# Updated run_absa.py with English print statements and without fp16 support

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch run FinABSA-Longer on large datasets.

Features:
- Loads input CSV containing a 'snippet' column
- Runs FinABSA-Longer (GPU batched) to produce:
    absa_label, absa_logits, absa_probs
- Supports checkpointing and resume
- Designed for tmux long-running stability

Usage:
    python run_absa.py --input input.csv --output output.csv --batch-size 8 --checkpoint-rows 4000

Run inside tmux:
    tmux
    python run_absa.py --input ... --output ...
    # Detach with: Ctrl+B then D
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------------------------------------
# 1. FinABSALonger Model Wrapper
# -----------------------------------------------------------

class FinABSALonger:
    def __init__(
        self,
        ckpt_path: str = "amphora/FinABSA-Longer",
        max_input_length: int = 1024,
        max_gen_length: int = 32,
        num_beams: int = 4,
    ):
        """
        Simple wrapper for FinABSA-Longer. FP16 removed as requested.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FinABSALonger] Device: {self.device}")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        self.max_input_length = max_input_length
        self.max_gen_length = max_gen_length
        self.num_beams = num_beams

        self.label_tokens = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        self.label_token_ids = [
            self.tokenizer(t, add_special_tokens=False)["input_ids"][0]
            for t in self.label_tokens
        ]
        self.idx2label = {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"}

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "classification_output": None,
            "probs": None,
            "logits": None,
            "raw_output": None,
        }

    def analyze_batch(
        self,
        texts: List[Optional[str]],
        logit_flag: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference. Returns list of dicts matching the old analyze() format.
        """
        n = len(texts)
        results: List[Dict[str, Any]] = [self._empty_result() for _ in range(n)]

        valid_indices = []
        valid_texts = []
        for i, t in enumerate(texts):
            if isinstance(t, str) and t.strip():
                valid_indices.append(i)
                valid_texts.append(t)

        if not valid_texts:
            return results

        enc = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_length=self.max_gen_length,
                num_beams=self.num_beams,
                output_scores=logit_flag,
                return_dict_in_generate=logit_flag,
            )

        if logit_flag:
            sequences = out["sequences"]
            scores = out["scores"]
        else:
            sequences = out
            scores = None

        for batch_idx, seq in enumerate(sequences):
            raw_output = self.tokenizer.decode(seq, skip_special_tokens=True)

            if len(seq) < 4:
                cls_tok = None
            else:
                cls_token_id = int(seq[-4])
                cls_tok = self.tokenizer.decode([cls_token_id]).strip()

            base_res = {
                "classification_output": cls_tok,
                "probs": None,
                "logits": None,
                "raw_output": raw_output,
            }

            if (not logit_flag) or (cls_tok is None) or (scores is None):
                res = base_res
            else:
                label_token_index = len(seq) - 4
                step_idx = label_token_index - 1

                if step_idx < 0 or step_idx >= len(scores):
                    res = base_res
                else:
                    step_logits_full = scores[step_idx][batch_idx]
                    sentiment_logits = torch.tensor(
                        [step_logits_full[tid].item() for tid in self.label_token_ids],
                        device=step_logits_full.device,
                    )
                    probs_tensor = F.softmax(sentiment_logits, dim=-1)

                    logits_dict = {
                        self.idx2label[i]: float(sentiment_logits[i]) for i in range(3)
                    }
                    probs_dict = {
                        self.idx2label[i]: float(probs_tensor[i]) for i in range(3)
                    }

                    res = {
                        "classification_output": cls_tok,
                        "probs": probs_dict,
                        "logits": logits_dict,
                        "raw_output": raw_output,
                    }

            orig_idx = valid_indices[batch_idx]
            results[orig_idx] = res

        return results


# -----------------------------------------------------------
# 2. Resume + Load Utilities
# -----------------------------------------------------------

def load_or_init_df(input_path: str, output_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df_in = pd.read_csv(input_path)

    if os.path.exists(output_path):
        print(f"[INFO] Existing output found. Resuming from: {output_path}")
        df_out = pd.read_csv(output_path)
        if len(df_out) != len(df_in):
            raise ValueError(
                f"Input and output CSV length mismatch: {len(df_in)} vs {len(df_out)}"
            )
        df = df_out
    else:
        df = df_in.copy()
        if "absa_label" not in df.columns:
            df["absa_label"] = None
        if "absa_logits" not in df.columns:
            df["absa_logits"] = None
        if "absa_probs" not in df.columns:
            df["absa_probs"] = None

    if "snippet" not in df.columns:
        raise KeyError("Input CSV must contain a 'snippet' column.")

    return df


def find_start_index(df: pd.DataFrame) -> int:
    mask_unfinished = df["absa_label"].isna() | (df["absa_label"] == "")
    unfinished = df.index[mask_unfinished].tolist()
    return unfinished[0] if unfinished else len(df)


# -----------------------------------------------------------
# 3. Main Runner
# -----------------------------------------------------------

def run_absa(
    input_path: str,
    output_path: str,
    batch_size: int = 8,
    checkpoint_rows: int = 4000,
    max_rows: Optional[int] = None,
):
    df = load_or_init_df(input_path, output_path)
    n_total = len(df)

    if max_rows is not None:
        n_total = min(n_total, max_rows)
        df = df.iloc[:n_total].copy()

    start_idx = find_start_index(df)
    if start_idx >= n_total:
        print("[INFO] All rows already processed.")
        return

    print(f"[INFO] Total rows: {n_total}, starting from index {start_idx}")
    print(f"[INFO] Batch size: {batch_size}, checkpoint every {checkpoint_rows} rows")

    absa = FinABSALonger(
        ckpt_path="amphora/FinABSA-Longer",
        max_input_length=1024,
        max_gen_length=32,
        num_beams=4,
    )

    processed_since_ckpt = 0

    for start in tqdm(range(start_idx, n_total, batch_size), desc="Running FinABSA-Longer"):
        end = min(start + batch_size, n_total)
        batch_idx = range(start, end)
        batch_texts = df.loc[batch_idx, "snippet"].tolist()

        try:
            batch_results = absa.analyze_batch(batch_texts, logit_flag=True)
        except torch.cuda.OutOfMemoryError:
            print("\n[ERROR] CUDA OOM during batch. Reduce batch-size.")
            return

        for i, idx in enumerate(batch_idx):
            res = batch_results[i]
            df.at[idx, "absa_label"] = res.get("classification_output")
            logits = res.get("logits")
            probs = res.get("probs")
            df.at[idx, "absa_logits"] = json.dumps(logits) if logits is not None else None
            df.at[idx, "absa_probs"] = json.dumps(probs) if probs is not None else None

        processed_since_ckpt += (end - start)

        if processed_since_ckpt >= checkpoint_rows:
            print(f"[INFO] Checkpoint: writing to {output_path}")
            df.to_csv(output_path, index=False)
            processed_since_ckpt = 0

    print(f"[INFO] Finished. Writing final output to {output_path}")
    df.to_csv(output_path, index=False)

    del absa
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] Done.")

# -----------------------------------------------------------
# 4. CLI
# -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run FinABSA-Longer on a CSV file")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path (will be overwritten during checkpoints)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for GPU inference")
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=4000,
        help="Write checkpoint every N processed rows",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally process only the first N rows (debugging)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_absa(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        checkpoint_rows=args.checkpoint_rows,
        max_rows=args.max_rows,
    )
