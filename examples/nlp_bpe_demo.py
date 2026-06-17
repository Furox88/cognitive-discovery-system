"""End-to-end demo of cds.nlp: train a BPE tokenizer and embed a sentence.

Trains a small BPE vocabulary on a short corpus (so the example runs
in <1s), encodes a sample sentence, then passes the resulting token
ids through the token + positional embedding pipeline. The output is
the per-position vector for each token — exactly what the (future)
attention block will consume in Sprint 2.

Run::

    python examples/nlp_bpe_demo.py
"""

from __future__ import annotations

from cds.nlp import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
    train_bpe,
)


def main() -> None:
    # 1. Tiny corpus — a handful of repeated words so BPE has clear
    #    frequency signals to merge on.
    corpus = (
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox jumps over the lazy dog "
        "she sells seashells by the seashore "
        "she sells seashells by the seashore "
    )

    # 2. Train the tokenizer.
    tokenizer = train_bpe(corpus, vocab_size=80, min_frequency=2)
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"merges learned: {len(tokenizer.merges)}")
    if tokenizer.merges:
        print("first three merges:")
        for merge in tokenizer.merges[:3]:
            print(f"  {merge.pair!r} -> {merge.new_token!r}")

    # 3. Encode a sentence.
    text = "the quick brown fox"
    ids = tokenizer.encode(text)
    print(f"\nencoded {text!r}:")
    print(f"  ids:    {ids}")
    print(f"  tokens: {[tokenizer.id_to_token[i] for i in ids]}")
    print(f"  decoded: {tokenizer.decode(ids)!r}")

    # 4. Round-trip sanity check.
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"round-trip failed: {decoded!r} != {text!r}"
    print("\nround-trip OK")

    # 5. Embedding pipeline.
    d_model = 16
    table = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    pe = PositionalEncoding(max_len=len(ids) + 4, d_model=d_model)
    tokens = table.forward(ids)
    combined = add_positional(tokens, pe)
    print(f"\nembedding shape: {len(combined)} x {len(combined[0])}")
    print(f"first token position vector (first 4 dims):")
    print(f"  token:  {tokens[0][:4]}")
    print(f"  pos:    {pe.forward(1)[0][:4]}")
    print(f"  sum:    {combined[0][:4]}")
    # add_positional is just element-wise sum — verify.
    for j in range(4):
        assert abs(combined[0][j] - (tokens[0][j] + pe.forward(1)[0][j])) < 1e-12

    # 6. Save / load round-trip.
    tokenizer.save("examples/_demo_tokenizer.json")
    loaded = type(tokenizer).load("examples/_demo_tokenizer.json")
    assert loaded.encode(text) == ids
    print("\nsave / load round-trip OK")


if __name__ == "__main__":
    main()