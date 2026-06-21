# Yeni Topluluk Post Taslakları

> Her subreddit için ayrı angle. Tüm metinler **em dash içermiyor**
> (LLM kokusu olmaması için). Görsel gerektirenler için GIF/MP4 hazır:
> `assets/cds_promo.gif` (5.44MB), `.mp4` (0.20MB), `.webm` (0.13MB).
>
> Son güncelleme: 2026-06-21

---

## 1) r/dataisbeautiful (~20M) - EN BÜYÜK FIRSAT  [DASHBOARD ILE GUNCEL]

> Bu subreddit **OC (Original Content)** ister, görsel + veri odaklı.
> GIF yerine artik **6 panel Monte Carlo dashboard PNG** var:
> `assets/monte_carlo_dashboard.png` (359 KB).
>
> Dashboard tamamen **gercek cds.montecarlo API**'lerinden uretilmistir:
> estimate_pi, buffon_needle, mc_integrate, random_walk_2d. Elle uydurulmus
> veri YOK. Bu DiB icin kritik -- "OC" etiketinin ici dolu olmali.
>
> 6 panel:
>   1. pi scatter (inside/outside + quarter circle)
>   2. convergence (running estimate vs N, +-1 sigma band)
>   3. Buffon's needle (P(crossing) -> pi)
>   4. error decay (log-log, O(1/sqrt N) reference)
>   5. MC integration (sin(x) over [0, pi])
>   6. 2D random walk (multiple trajectories)
>
> Post tipi: **image post** (PNG'yi Reddit'e yukle), metin govdesi kisa.

### Baslik

**Six views on Monte Carlo estimation, in pure Python [OC]**

### Govde

Hi everyone,

Sharing a visualization of Monte Carlo methods I put together. All six panels
are generated from a single pure-Python module I wrote (no NumPy, no SciPy,
no compiled extensions), so every number you see comes from actual code
running the algorithms, not hand-plotted data.

What's in each panel:

- **1 - pi via unit square.** 2,000 random points; the fraction inside the
  quarter circle times 4 converges to pi.
- **2 - convergence.** Running estimate vs sample count (log-x), with a
  +-1 sigma band. It tightens the way you'd expect from the central limit
  theorem.
- **3 - Buffon's needle.** Dropping needles on parallel lines, the crossing
  probability gives pi via 2L / (D . P). Slower convergence than the unit
  square, but a nice independent estimator.
- **4 - error decay.** |estimate - pi| on log-log. The dashed line is the
  O(1/sqrt N) theoretical rate, and the observed error tracks it closely.
- **5 - MC integration.** Integral of sin(x) over [0, pi] by uniform random
  sampling. True value is 2; the MC estimate sits right on it.
- **6 - 2D random walk.** Six independent diffusion trajectories, 500 steps
  each, showing the spread you get from symmetric random motion.

The point of doing it in pure Python isn't speed, it's readability: every
estimator above is a short function you can open and read top to bottom,
which is what made it easy to visualize each one honestly.

Tools: Python + matplotlib. Repo with the actual implementations:
https://github.com/Furox88/cognitive-discovery-system

Happy to dig into how any of the panels were computed. Which estimator
surprised you, or which would you want to see at higher sample counts?

---

## 2) r/learnpython (~600K) - EN GÜVENLI ILK DENEME

> Text post. Öğretici angle. Self-promo toleransı yüksek.
> Soruyla bitir (feedback ister).

### Başlık

**I wrote a pure-Python scientific library to teach myself FFT, RK4 and quantum circuits without NumPy. Sharing in case it helps other learners.**

### Gövde

Hi r/learnpython,

A while back I wanted to actually understand how FFT, RK4 ODE solvers and basic quantum circuits work under the hood, not just call them from NumPy or SciPy. So I built a small library where every algorithm is written from scratch in plain Python.

A few things I learned along the way that might help others:

- **Radix-2 FFT** clicks once you see the recursive split into even and odd samples. Writing it out made the O(N log N) actually make sense to me, instead of memorizing it.
- **RK4** is just four slope estimates averaged with clever weights. Seeing it in 20 lines of Python beats any diagram I'd seen.
- **Partial-pivoting LU** demystified why "just invert the matrix" is naive. The pivoting logic is where the real subtlety lives.
- **Bell state entanglement** becomes concrete when you implement measure and see the correlation in the outputs.

The library ended up at 17 modules (signals, stats, ML, graphs, NLP, quantum and more) and I kept everything dependency-light so it runs anywhere Python runs.

Repo if useful: https://github.com/Furox88/cognitive-discovery-system

Curious whether anyone else here learns better by reimplementing things from scratch, or if I'm just weird. Any algorithms that clicked for you only once you wrote them out?

---

## 3) r/SideProject (~700K) - SELF-PROMO DOSTU

> Showcase angle. Rahat, samimi ton. GIF eklenebilir.

### Başlık

**Show SideProject: A pure-Python scientific computing library (17 modules, no NumPy)**

### Gövde

Hey everyone,

Wanted to share a side project I've been building. It's a scientific computing library written entirely in pure Python, with zero heavy dependencies. No NumPy, no SciPy, no compiled extensions, no BLAS toolchain.

What's inside:

- Quantum simulation (Bell and GHZ states, entanglement detection)
- Signal processing (radix-2 FFT, convolution via the Convolution Theorem)
- Numerical methods (LU/QR/Cholesky, RK4, Gauss-Legendre quadrature)
- Statistics (t-test, chi-square, ANOVA, regression)
- Machine learning from scratch (MLP with Adam)
- NLP basics (BPE tokenizer, multi-head attention)
- Plus graphs, Monte Carlo, symbolic math, hypothesis scaffolding

Why pure Python? Two reasons. First, readable source you can actually learn from, every algorithm is right there in plain code. Second, it runs in edge environments where installing NumPy or a C toolchain isn't practical.

It's on PyPI: `pip install cognitive-discovery-system`

Repo: https://github.com/Furox88/cognitive-discovery-system
Docs: https://Furox88.github.io/cognitive-discovery-system/

Not trying to compete with NumPy. Just a learning-first, dependency-light alternative. Would love honest feedback on whether the "one umbrella for everything" approach reads as useful or unfocused.

---

## 4) r/LocalLLaMA (~600K) - AI/TEKNIK KITLE

> NLP angle. Bu kitle BPE, attention, MiniGPT'den anlar.
> Teknik detay ver, yoksa "basit toy" derler.

### Başlık

**Implemented BPE tokenizer and multi-head attention from scratch in pure Python (no PyTorch). Sharing the implementations.**

### Gövde

Sharing something that might be useful to folks here who like to understand what's under the hood of tokenizers and attention, rather than treating them as black boxes.

As part of a larger pure-Python scientific library, I implemented from scratch:

- **Byte-Pair Encoding tokenizer** with merge rule learning, vocabulary building and deterministic encoding/decoding. No regex pre-tokenizer hacks, just the core BPE loop with proper rank tracking.
- **Multi-head attention** with scaled dot-product, learnable Q/K/V projections and causal masking. Plain Python lists and math, no tensors.
- A small **MiniGPT-style demo** that wires embedding, attention and a feed-forward block together so you can see the full forward pass in readable code.

Why bother, when PyTorch exists? Two reasons that made it worth it for me:

1. Reading the actual implementation, line by line, taught me more about why attention works than any blog post. The Q*K^T scaling by sqrt(d_k), the softmax over masked rows, the per-head independence, all of it makes sense once you see it in plain loops.
2. It runs anywhere Python runs. No CUDA, no wheel compatibility headaches. Useful for embedded or constrained environments.

The NLP module sits alongside quantum, FFT, ODE solvers and stats in the same library, all pure Python.

Repo: https://github.com/Furox88/cognitive-discovery-system

If anyone here has implemented attention or BPE from scratch themselves, I'd genuinely value feedback on whether the structure matches how you'd do it, or what you'd do differently.

---

## Zamanlama ve sıra

| Subreddit | Ne zaman | Not |
|-----------|----------|-----|
| r/learnpython | 1. | En güvenli, yeni hesap için |
| r/SideProject | +2 gün sonra | Self-promo dostu |
| r/dataisbeautiful | +2 gün sonra | GIF'i ekle, OC etiketi |
| r/LocalLLaMA | +2 gün sonra | En teknik, dikkatli ol |

Aynı gün iki yere post atma. En az 2-3 gün ara.
