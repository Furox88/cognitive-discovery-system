# Reddit Master Post Template

Tek bir ana metin. Her subreddit için "angle" paragrafı değiştiriliyor,
gerisi ortak. Tüm bilgiler gerçek (README + docs/benchmarks.md'den).

---

## MASTER POST (tam metin)

Hey everyone,

I've been working on an open-source scientific computing library called
cognitive-discovery-system (CDS). It's written entirely in pure Python —
no NumPy, no SciPy, no compiled C extensions, no BLAS/LAPACK binaries.

[ANGLE PARAGRAFI — aşağıdaki kutudan seç]

The library covers 17 modules:

- Quantum: single/multi-qubit circuits, Bell/GHZ states, entanglement
  detection, measurement via probabilistic sampling
- Signals: radix-2 FFT, 2D FFT, convolution via Convolution Theorem
- Linear algebra: LU (partial pivoting), QR, Cholesky, eigenvalues
- ODE solvers: Euler, midpoint, RK4
- Numerical integration: Simpson, Romberg, Gauss-Legendre
- Optimization: gradient descent, Newton's method, Adam
- Statistics: t-test, chi-square, ANOVA, regression
- Probability: discrete/continuous distributions, sampling
- Monte Carlo: simulation, integration, estimation
- Graph algorithms: Dijkstra, Kruskal MST, topological sort
- Symbolic math: differentiation, expression trees
- ML: neural network from scratch (MLP + Adam)
- NLP: BPE tokenizer, multi-head attention, Transformer, MiniGPT
  (with a small scalar autograd)

Honest take: it's not faster than NumPy. It can't be. The point is a
toolkit you can read end-to-end — every algorithm implemented from
scratch, line by line. Good for learning, prototyping, and environments
where a C toolchain isn't available.

For real work, use NumPy/SciPy. For benchmarks against NumPy:
https://furox88.github.io/cognitive-discovery-system/benchmarks/

Quick install:

    pip install cognitive-discovery-system

Repo: https://github.com/Furox88/cognitive-discovery-system
Docs: https://furox88.github.io/cognitive-discovery-system/

Curious what you think — what would make something like this more
useful to you?

---

## ANGLE PARAGRAFLARI (her subreddit için farklı)

### Genel / side project toplulukları
(r/SideProject, r/IMadeThis, r/coolgithubprojects, r/opensource)

```
I built this as a side project to understand how these algorithms
actually work under the hood. Wanted to see if I could write LU, FFT,
RK4, a small Transformer — all from scratch, in readable Python.
```

### Eğitim odaklı topluluklar
(r/learnpython, r/PythonLearning, r/learnmachinelearning)

```
This started as a learning exercise — I wanted to read every line of
the math I was using. Turned into a 17-module library meant for
teaching: each algorithm is readable, plain Python, no black boxes.
```

### Matematik/bilim toplulukları
(r/math, r/3blue1brown, r/compsci, r/algorithms)

```
The design principle is "intelligence over brute force" — where pure
Python can't win on raw loops, it picks better algorithms: O(N log N)
FFT instead of O(N^2) DFT, O(N^3) LU instead of naive O(N!) approaches,
O(1) quantum measurement via probabilistic sampling instead of
shot-by-shot matrix ops.
```

### Python topluluğu (warm-up sonrası)
(r/Python)

```
Pure-Python scientific computing, zero dependencies. Useful when you
need to run scientific code somewhere NumPy can't go, or when you want
a readable reference implementation instead of a black box.
```

### Görselleştirme toplulukları
(r/dataisbeautiful)

```
Part of the project includes ASCII visualizations and a dashboard for
exploring the modules — thought it might be interesting to people here
who like data viz from scratch.
```

---

## SUBREDDIT BAZLI NOTLAR

- r/ProgrammingLanguages: ATMA. LLM-asistan kuralına takılır.
- r/Python: Sadece Salı "Show & Tell" thread'inde, veya warm-up sonrası
  stand-alone post. Yeni hesapta modfilter'a takılır.
- r/quantumcomputing / r/MachineLearning: Mod onayı gerekir, 24 saat
  bekle, onaylanmazsa modlara mesaj at.
- r/learnpython / r/SideProject / r/IMadeThis: En düşük sürtünme.
- Aynı gün iki subreddit'e post ATMA — en az 2-3 gün ara.
- Her postun sonunda soru var (engagement hook): "what would make this
  more useful to you?" — bu yorum çekmeyi artırır.

---

## BAŞLIK ÖNERİLERİ (subreddit'e göre)

- Genel: "I built a pure-Python scientific computing library - no NumPy,
  no SciPy"
- Eğitim: "A from-scratch scientific library for learning how FFT, LU,
  RK4 actually work"
- Matematik: "Pure-Python implementations of FFT, LU decomposition,
  ODE solvers - algorithmic choices over brute force"
- Side project: "Side project: 17-module scientific computing library in
  pure Python, zero dependencies"

---

## ASLA

- "Star verin" / "destekleyin" deme
- LLM kullanmadım deme (gerçek değil, ban sebebi)
- Aynı postu aynı gün birden fazla subreddit'e atma
- AI yazısı gibi mükemmel markdown kullanma — düz metin
