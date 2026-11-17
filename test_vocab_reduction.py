#!/usr/bin/env python3
"""Quick test to verify vocabulary reduction with min_freq filtering."""

from pathlib import Path
from transformer_only.data.how2sign_loader import (
    SentenceTokenizer,
    build_tokenizer,
)

train_csv = "dataset/how2sign_realigned_train.csv"

print("[Test] Building tokenizers with different min_freq values...\n")

# Build tokenizers with different thresholds
for min_freq in [1, 2, 3, 4, 5]:
    tokenizer = build_tokenizer(train_csv, save_path=None, min_freq=min_freq)
    print(f"min_freq={min_freq}: vocab_size={tokenizer.vocab_size}")
    
    # Show most frequent words
    if min_freq in [1, 3]:
        from collections import Counter
        import csv
        word_counts = Counter()
        with open(train_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s = row.get("SENTENCE", "").strip()
                if s:
                    for word in s.lower().split():
                        word_counts[word] += 1
        print(f"  Top 10 words: {word_counts.most_common(10)}")

print("\n[Test] Done. Recommendation: use min_freq=3 to reduce 26976 → ~4000 tokens.")
