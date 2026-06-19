"""End-to-end demo of cds.nlp: train MiniGPT on Shakespeare.

The full pipeline:
1. Load the Shakespeare excerpt from :mod:`cds.nlp.data`.
2. Train a tiny MiniGPT (one causal block, d_model=32) with
   :func:`cds.nlp.training.train_step` + Adam.
3. Sample text from the trained model and print it.

Run::

    python examples/nlp_mini_gpt_demo.py
"""

from __future__ import annotations

from cds.nlp.autograd.tensor import Tensor
from cds.nlp.data import TEXT, decode, encode, vocab_size
from cds.nlp.model import MiniGPT
from cds.nlp.optim import Adam
from cds.nlp.training import train_step

# MiniGPT.parameters() returns list[Parameter], a subclass of Tensor.
# Adam.params is list[Tensor]; the assignment is sound at runtime but
# mypy flags it because list is invariant. We import Tensor for typing.



def main() -> None:
    # 1. Encode the corpus.
    ids = encode(TEXT)
    print(f"corpus: {len(ids)} chars, vocab_size={vocab_size}")

    # 2. Build the model.
    d_model = 32
    n_heads = 2
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=64,
        max_len=32,
        seed=42,
    )
    n_params = sum(1 for _ in model.parameters())
    print(f"model: d_model={d_model}, n_heads={n_heads}, ~{n_params} parameters")

    # 3. Train: slide a window of length ``T`` over the corpus, predict
    #    the next char, update the model.
    T = 24
    # Adam expects list[Tensor]; MiniGPT.parameters() returns list[Parameter],
    # a Tensor subclass. Cast widens the invariant list for mypy.
    params: list[Tensor] = model.parameters()  # type: ignore[assignment]
    optimiser = Adam(params=params, lr=0.005)
    n_steps = 200
    losses: list[float] = []
    for step in range(n_steps):
        # Pick a random starting offset, take a T-length window, target = next char.
        import random

        start = random.randint(0, max(0, len(ids) - T - 1))
        x = ids[start : start + T]
        y = ids[start + T]
        loss = train_step(model.forward, x, y, optimiser)
        losses.append(loss)
        if step % 20 == 0:
            avg = sum(losses[-20:]) / min(20, len(losses))
            print(f"  step {step:3d}: loss={avg:.4f}")

    # 4. Sample.
    print("\n--- sample ---")
    prompt = encode("All the ")
    out = model.sample(prompt, n_tokens=120)
    print(decode(out))


if __name__ == "__main__":
    main()
