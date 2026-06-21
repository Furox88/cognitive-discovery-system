# Video Post Taslakları (cds_promo.mp4 ile)

> Düz text değil, **video** ile. Video içeriği (doğrulandı):
> CDS wordmark + orbit emblem, "pure Python, no NumPy/SciPy" sloganı,
> terminalde `bell_state` / `is_entangled` / `measure_shots` demosu,
> 17 modules / 1244 tests / 100% coverage footer.
>
> Video: `assets/cds_promo.mp4` (0.20MB), `.gif` (5.44MB), `.webm` (0.13MB)
> Son güncelleme: 2026-06-21
> Tüm metinler em dash içermiyor (LLM kokusu olmaması için).

---

## Post etme notu (önce bunu oku)

Reddit'e video atarken:
- mp4'yi Reddit'in kendi video upload'ı ile yükle (link post degil).
- Bazı subreddit'ler video kabul etmez, o zaman GIF'i image post yap.
- r/dataisbeautiful OC (Original Content) etiketi ister, postta [OC] koy.

---

## 1) r/dataisbeautiful (~20M) - EN BÜYÜK FIRSAT  [DASHBOARD ONERILIR]

> NOT: dataisbeautiful icin artik **6 panel Monte Carlo dashboard PNG**
> oneriliyor (`assets/monte_carlo_dashboard.png`, 359 KB). DiB bu postu
> cok daha iyi karsilar cunku "veri odakli OC" tanimina tam uyuyor.
>
> Asagidaki VIDEO postu hala alternatif olarak kullanilabilir ama Once
> dashboard PNG'yi dene. Dashboard metni: `NEW_COMMUNITY_POSTS.md` icinde.

### Başlık

**A 17-module pure-Python scientific library, visualized [OC]**

### Gövde (videoya eşlik eden açıklama)

Short visualization of a project I've been building. It's a scientific computing library written entirely in pure Python, zero heavy dependencies. No NumPy, no SciPy, no compiled extensions.

The video cycles through the core idea and a live quantum demo in the terminal, Bell state entanglement measured over 1000 shots.

Modules shown cover quantum simulation, FFT, linear algebra, stats, optimization, NLP and structured hypothesis generation. Everything implemented from scratch in readable Python, not calling into BLAS or C.

Source if you want to read the actual implementations: https://github.com/Furox88/cognitive-discovery-system
Docs: https://Furox88.github.io/cognitive-discovery-system/

What stood out to you visually? Happy to dig into any of the algorithms behind the animation.

---

## 2) r/learnpython (~600K) - EN GÜVENLI

### Başlık

**Wrote a pure-Python scientific library (no NumPy) to teach myself FFT, RK4 and quantum circuits. Here's a quick visual.**

### Gövde

I wanted to actually understand how FFT, RK4 ODE solvers and basic quantum circuits work under the hood, not just call them from NumPy. So I built a library where every algorithm is written from scratch in plain Python.

The video shows the core pitch and a small live demo: a Bell state, entanglement check, then 1000 measurements split roughly 50/50 between 00 and 11.

A few things that clicked for me by writing them out:
- Radix-2 FFT makes sense once you see the even/odd split
- RK4 is four slope estimates averaged with clever weights
- Partial-pivoting LU is where the real subtlety of "invert a matrix" lives
- Bell state entanglement becomes concrete when you implement measure

Library is 17 modules now, all pure Python, runs anywhere Python runs.

Repo: https://github.com/Furox88/cognitive-discovery-system

Anyone else here learn better by reimplementing things from scratch? Which algorithm only made sense to you once you wrote it yourself?

---

## 3) r/SideProject (~700K) - SELF-PROMO DOSTU

### Başlık

**Show SideProject: A pure-Python scientific computing library, 17 modules, no NumPy (short demo video)**

### Gövde

Sharing a side project. It's a scientific computing library written entirely in pure Python, zero heavy dependencies. No NumPy, no SciPy, no compiled extensions.

The video runs through the core idea and a live terminal demo with a Bell state and entanglement detection.

What's inside:
- Quantum simulation (Bell/GHZ states, entanglement)
- Signal processing (radix-2 FFT, convolution)
- Numerical methods (LU/QR/Cholesky, RK4, Gauss-Legendre)
- Stats, ML from scratch (MLP with Adam), NLP basics (BPE, attention)
- Plus graphs, Monte Carlo, symbolic math, hypothesis scaffolding

Two reasons it exists: readable source you can learn from, and it runs in edge environments where NumPy or a C toolchain isn't practical.

PyPI: `pip install cognitive-discovery-system`
Repo: https://github.com/Furox88/cognitive-discovery-system
Docs: https://Furox88.github.io/cognitive-discovery-system/

Not trying to compete with NumPy. Would love honest feedback on whether the "one umbrella for everything" idea reads as useful or unfocused.

---

## 4) r/LocalLLaMA (~600K) - AI/TEKNIK KITLE

### Başlık

**BPE tokenizer and multi-head attention from scratch in pure Python, no PyTorch (demo video inside)**

### Gövde

For folks who like to understand what's actually happening inside tokenizers and attention rather than treating them as black boxes.

As part of a larger pure-Python scientific library I implemented from scratch:

- BPE tokenizer with merge rule learning, vocab building, deterministic encode/decode
- Multi-head attention with scaled dot-product, learnable Q/K/V, causal masking. Plain Python, no tensors
- A MiniGPT-style demo wiring embedding, attention and a feed-forward block together

The video shows the broader library pitch (the NLP module sits alongside quantum, FFT, ODE solvers and stats), plus a live terminal demo of quantum entanglement so you can see the kind of readability the whole thing is built around.

Why bother, when PyTorch exists:
1. Reading the implementation line by line taught me more about why attention works than any blog post. The sqrt(d_k) scaling, the masked softmax, per-head independence, all of it clicks once you see it in plain loops
2. Runs anywhere Python runs. No CUDA, no wheel headaches

Repo: https://github.com/Furox88/cognitive-discovery-system

If anyone here has implemented attention or BPE from scratch, I'd value feedback on whether the structure matches how you'd do it.

---

## Zamanlama

| Subreddit | Ne zaman | Not |
|-----------|----------|-----|
| r/learnpython | 1. | En güvenli, yeni hesap için |
| r/SideProject | +2 gün sonra | Self-promo dostu |
| r/dataisbeautiful | +2 gün sonra | Video + [OC] etiketi |
| r/LocalLLaMA | +2 gün sonra | En teknik kitle |

Aynı gün iki yere post atma. En az 2-3 gün ara.
