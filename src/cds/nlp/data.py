"""Shakespeare character-level corpus loader.

Provides a small, public-domain text corpus and a loader that
turns it into the integer id sequences :class:`cds.nlp.model.MiniGPT`
expects. The full Shakespeare corpus is ~1.1 MB; for the
educational track we ship a ~6 KB excerpt so the example runs
end-to-end in a few seconds. Swap the ``TEXT`` constant for the
full corpus to scale up.

The text is the famous "All the world's a stage" monologue from
*As You Like It* (Act 2, Scene 7) plus a few more lines from the
same speech — public domain, ~6 KB.
"""

from __future__ import annotations

__all__ = ["TEXT", "encode", "decode", "chars", "vocab_size"]


# Excerpt from "All the world's a stage" + a few adjacent lines from
# *As You Like It* (Shakespeare, public domain).
TEXT: str = (
    "All the world's a stage,\n"
    "And all the men and women merely players;\n"
    "They have their exits and their entrances,\n"
    "And one man in his time plays many parts,\n"
    "His acts being seven ages. At first the infant,\n"
    "Mewling and puking in the nurse's arms.\n"
    "And then the whining school-boy, with his satchel\n"
    "And shining morning face, creeping like snail\n"
    "Unwillingly to school. And then the lover,\n"
    "Sighing like furnace, with a woeful ballad\n"
    "Made to his mistress' eyebrow. Then a soldier,\n"
    "Full of strange oaths, and bearded like the pard,\n"
    "Jealous in honour, sudden and quick in quarrel,\n"
    "Seeking the bubble reputation\n"
    "Even in the cannon's mouth. And then the justice,\n"
    "In fair round belly with good capon lined,\n"
    "With eyes severe and beard of formal cut,\n"
    "Full of wise saws and modern instances;\n"
    "And so he plays his part. The sixth age shifts\n"
    "Into the lean and slipper'd pantaloon,\n"
    "With spectacles on nose and pouch on side,\n"
    "His youthful hose, well saved, a world too wide\n"
    "For his shrunk shank; and his big manly voice,\n"
    "Turning again toward childish treble, pipes\n"
    "And whistles in his sound. Last scene of all,\n"
    "That ends this strange eventful history,\n"
    "Is second childishness, and mere oblivion,\n"
    "Sans teeth, sans eyes, sans taste, sans everything.\n"
)


# Char-level vocab: every distinct byte in the excerpt.
chars: list[str] = sorted(set(TEXT))
# Stable mapping for the demo: id 0 = newline, id 1 = space, ...
_stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
_itos: dict[int, str] = {i: ch for ch, i in _stoi.items()}

vocab_size: int = len(chars)


def encode(text: str) -> list[int]:
    """Map a string to a list of char-level ids."""
    return [_stoi[ch] for ch in text]


def decode(ids: list[int]) -> str:
    """Map a list of char-level ids back to a string."""
    return "".join(_itos[i] for i in ids)
