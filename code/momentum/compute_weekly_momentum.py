"""Compute weekly momentum factor from daily returns CSV.

Outputs a CSV with columns: TICKER,week_date,factor

Default momentum: past 12 weeks cumulative return excluding the most recent week (common convention).

Usage:
    python compute_weekly_momentum.py \
        --input ./s&p500ret.csv \
        --output ./weekly_momentum.csv \
        --lookback 12 --exclude 1
"""
from pathlib import Path
import argparse
import pandas as pd


def compute_weekly_momentum(df: pd.DataFrame, lookback: int = 12, exclude_recent: int = 1) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['TICKER', 'date'])

    # Map each date to the week-ending (Friday) date
    # Use week period anchored to Friday and take the period end timestamp
    df['week_date'] = df['date'].dt.to_period('W-FRI').apply(lambda p: p.end_time.date())

    # Compute weekly return per ticker-week: product(1+RET)-1
    weekly = (
        df.groupby(['TICKER', 'week_date'], as_index=False)
        .agg({'RET': lambda x: (1 + x).prod() - 1})
        .rename(columns={'RET': 'weekly_ret'})
    )

    weekly = weekly.sort_values(['TICKER', 'week_date'])

    # Define function to compute rolling momentum per ticker
    def _rolling_mom(s: pd.Series) -> pd.Series:
        # shift to exclude the most recent weeks (e.g., exclude_recent=1)
        shifted = s.shift(exclude_recent)
        # compute rolling product of (1+shifted) over `lookback` periods
        prod = shifted.add(1).rolling(window=lookback, min_periods=lookback).apply(lambda arr: arr.prod(), raw=True)
        return prod - 1

    weekly['factor'] = weekly.groupby('TICKER')['weekly_ret'].transform(_rolling_mom)

    # Keep only rows where factor is available
    out = weekly[['TICKER', 'week_date', 'factor']].dropna().reset_index(drop=True)
    # Ensure week_date is ISO format string for CSV
    out['week_date'] = pd.to_datetime(out['week_date']).dt.strftime('%Y-%m-%d')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=str, default='s&p500ret.csv', help='Input CSV path (daily returns)')
    p.add_argument('--output', '-o', type=str, default='weekly_momentum.csv', help='Output CSV path')
    p.add_argument('--lookback', type=int, default=12, help='Number of weeks for momentum lookback')
    p.add_argument('--exclude', type=int, default=1, help='Number of most recent weeks to exclude')
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to this script's directory
        input_path = Path(__file__).resolve().parent / input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(input_path)

    # Validate required columns
    for c in ('date', 'TICKER', 'RET'):
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in input CSV")

    out = compute_weekly_momentum(df, lookback=args.lookback, exclude_recent=args.exclude)

    out_path = Path(args.output)
    # If output path is relative, place it relative to script directory
    if not out_path.parent.exists():
        out_path = Path(__file__).resolve().parent / out_path

    out.to_csv(out_path, index=False)
    print(f"Wrote weekly momentum to: {out_path}")


if __name__ == '__main__':
    main()
