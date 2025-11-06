import torch
import flair
from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.functional as F

class ABSA():
    def __init__(self, 
                 ckpt_path="amphora/FinABSA",
                 NER_tag_list = ['ORG']
                 ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flair.device = self.device 
        print(f"Using device: {self.device}")

        self.ABSA = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
        self.ABSA.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.tagger = SequenceTagger.load('ner')

        self.NER_tag_list = NER_tag_list

    def run_absa(self,input_str):
        tgt_entities = self.retrieve_target(input_str)
        output = {e : self.run_single_absa(input_str,e) for e in tgt_entities}
        return output

    def run_single_absa(self,input_str,tgt):
        input_str = input_str.replace(tgt, '[TGT]')
        # Add truncation to prevent OOM from long sequences
        input = self.tokenizer(input_str, return_tensors='pt',
                              truncation=True, max_length=512)

        # Log if text was truncated
        token_length = input['input_ids'].shape[1]
        if token_length >= 512:
            self.logger.warning(f"Text truncated to 512 tokens for entity '{tgt}'")
        else:
            self.logger.debug(f"Token length: {token_length} for entity '{tgt}'")

        input = {k: v.to(self.device) for k, v in input.items()}

        # Generate output without gradient tracking
        output = self.ABSA.generate(
                                    **input,
                                    max_length=20,
                                    output_scores=True,
                                    return_dict_in_generate=True
                                    )

        # Extract needed values and move to CPU immediately
        classification_output = self.tokenizer.convert_ids_to_tokens(
                                                    int(output['sequences'][0][-4].cpu())
                                                    )
        # Move logits to CPU and convert to Python floats to free GPU memory
        logits = F.softmax(output['scores'][-4][:,-3:].cpu(),dim=1)[0]

        result = {
                "classification_output": classification_output,
                "logits":
                {
                    'positive': float(logits[0]),
                    'negative': float(logits[1]),
                    'neutral':  float(logits[2])
                }
        }

        # Explicitly delete large tensors to free GPU memory
        del output
        del input
        del logits

        return result

    def retrieve_target(self,input_str):
        sentence = Sentence(input_str)
        with torch.no_grad():
            self.tagger.predict(sentence)
        entities = [entity.text for entity in sentence.get_spans('ner') if entity.tag in self.NER_tag_list]
        return entities

    def clear_memory(self):
        """Clear GPU/MPS memory cache to prevent OOM errors during batch processing"""
        if self.device.type == 'cuda':
            # Log memory stats before clearing
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.debug(f"CUDA memory before clear - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

            torch.cuda.empty_cache()

            # Log memory stats after clearing
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            self.logger.debug(f"CUDA memory after clear - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB")

        elif self.device.type == 'mps':
            torch.mps.empty_cache()
            self.logger.debug("MPS memory cache cleared")
