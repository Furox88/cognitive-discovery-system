# Reddit Launch Posts — Final Drafts

> **Amaç:** Gerçek değer sunan proje tanıtımı. Codex for OSS için **star rica yok** —
> bunun yerine teknik farkı, nasıl yapıldığını ve neden ilginç olduğunu anlatıyoruz.
> Bu yaklaşım, Reddit kültürüne uygun ve gerçek kullanıcı çeker.

---

## Post 1 — r/Python (ana hedef)

### Başlık (Title)

**I built a pure-Python scientific computing library — no NumPy/SciPy, 17 modules, 1,244 tests, 100% coverage**

### Gövde (Body)

---

Hi r/Python,

I've been working on a project I wanted to share and get feedback on. It's called **Cognitive Discovery System (CDS)** — a scientific computing library written in **pure Python** with **zero heavy dependencies** (no NumPy, no SciPy, no compiled extensions).

### Why pure Python?

The goal wasn't to outperform NumPy — it can't, and it shouldn't try. The goal was:

- **Readable source** you can actually learn from (every algorithm is implemented from scratch)
- **Edge runtime** — runs anywhere Python runs, no BLAS/C-Fortran toolchain
- **A single umbrella** for math + physics + stats + ML + signals + quantum, instead of 6+ separate libraries

### What's inside (17 modules)

| Area | Examples |
|------|----------|
| Quantum | Single & multi-qubit circuits, Bell/GHZ states, entanglement detection |
| Signals | O(N log N) FFT, convolution via Convolution Theorem, power spectrum |
| Optimization | Gradient descent, Newton's method, Adam |
| Numerical | O(N³) LU/QR/Cholesky, RK4 ODE solver, Gauss-Legendre quadrature |
| Stats | t-test, chi-square, ANOVA, Pearson, linear regression |
| ML | Neural network from scratch (MLP, Adam) |
| NLP | BPE tokenizer, multi-head attention, MiniGPT demo |
| Graphs | Dijkstra, Kruskal MST, topo sort |
| + Monte Carlo, symbolic math, knowledge graph, hypothesis generation |

### One design choice I'm proud of

Instead of naive shot-by-shot quantum sampling, I used **O(1) probabilistic sampling with explicit state collapse**. Similarly, the linear algebra uses partial-pivoting LU (O(N³)) instead of naive determinant expansion (O(N!)). The whole library is built around "smarter algorithms, not faster loops."

### Engineering side

- **1,244 tests, 100% statement + branch coverage**
- Multi-OS CI matrix, CodeQL, signed PyPI releases (OIDC)
- mkdocs site, CLI, interactive web dashboard

### Install

```bash
pip install cognitive-discovery-system
```

Repo: https://github.com/Furox88/cognitive-discovery-system
Docs: https://Furox88.github.io/cognitive-discovery-system/

### What I'd love feedback on

1. Is the "pure Python, no NumPy" pitch compelling, or does it sound like reinventing the wheel?
2. The library tries to be a broad umbrella — is that a strength (one package) or a weakness (unfocused)?
3. Any modules where the from-scratch implementation is particularly useful for learning?

Not trying to compete with NumPy — I'm genuinely curious whether there's a niche for a readable, dependency-light alternative. Happy to answer questions about any of the implementations.

---

## Post 2 — r/OpenAI (dikkatli, alaka odaklı)

> ⚠️ **Önemli:** r/OpenAI, bağımsız kütüphane tanıtımları için uygun bir yer değildir.
> Moderasyon riski yüksektir. Bu post, **OpenAI ile alakalı bir açı** bulmaya çalışır:
> "Codex kullanarak böyle bir kütüphane geliştirdim" — yani araç kullanımı üzerine.

### Başlık (Title)

**I used Codex/Claude-style AI coding agents to build a 17-module scientific library in 12 days — here's what worked and what didn't**

### Gövde (Body)

---

I wanted to share an experience report on using AI coding agents for a non-trivial software project — not a vibe-coded prototype, but a real library with 1,244 tests and 100% coverage.

### The project

Over the last ~12 days I built **Cognitive Discovery System** — a pure-Python scientific computing library (quantum simulation, FFT, ODE solvers, ML from scratch, stats, etc.) with zero NumPy/SciPy dependencies. It's now on PyPI.

### How AI agents were used (and not used)

**Worked well:**
- **Boilerplate and test scaffolding** — generating comprehensive test suites (1,244 tests). The agent was good at finding edge cases I'd miss.
- **Algorithm implementation** — translating mathematical descriptions (RK4, radix-2 FFT, LU decomposition) into clean Python. The from-scratch implementations are readable and correct.
- **CI/release plumbing** — OIDC PyPI publishing, CodeQL, multi-OS test matrix.

**Did NOT work well / required heavy human direction:**
- **Architectural decisions** — module boundaries, API ergonomics. The agent would happily produce *a* solution, but not always the *right* one. I had to drive the design.
- **"Intelligence over brute force" choices** — e.g., choosing O(1) probabilistic quantum sampling over shot-by-shot. This kind of optimization requires understanding the *why*, not just the *how*.
- **Avoiding scope creep** — a 17-module umbrella is already on the edge. Left to its own devices, the agent would happily add module #18, #19...

### The honest takeaway

AI agents are force multipliers for **breadth and consistency**, but the **quality bar is still set by the human**. The 100% coverage and clean APIs came from me insisting on them, not from the agent volunteering them.

If anyone is using Codex/agents for similar "real" engineering work (not just throwaway scripts), I'd love to hear how your experience compares.

Repo for reference: https://github.com/Furox88/cognitive-discovery-system

---

## 📋 Paylaşım öncesi kontrol listesi

- [ ] Postu **kopyalayıp Reddit'e yapıştırmadan önce** buradan oku
- [ ] r/Python postu: "Show and tell" kültürüne uygun, **sorularla bitiyor** (feedback rica)
- [ ] r/OpenAI postu: **AI kullanım deneyimi**ne odaklı, doğrudan proje tanıtımı değil
- [ ] **Star rica YOK** — her iki postta da "star verin" kelimesi geçmiyor
- [ ] Başlıklar clickbait değil, **sayılarla desteklenmiş** (17 modül, 1244 test, %100)
- [ ] Linkler: GitHub repo + docs sitesi
- [ ] r/Python postunu **ilk** paylaş, r/OpenAI'yi sonra (farklı kitle)

## ⏰ Zamanlama önerisi

- **r/Python**: Salı veya Çarşamba, **UTC 13:00–16:00** (ABD sabahı, Avrupa akşamı — en aktif pencere)
- **r/OpenAI**: r/Python postundan **2-3 gün sonra** (aynı anda değil, spam algısı oluşmasın)
- Post sonrası **ilk 1-2 saatte yorumlara cevap ver** — Reddit algoritması early engagement'i ödüllendirir

## 🚫 Yapma

- ❌ Aynı postu 5 subreddite cross-post yapma (spam algısı)
- ❌ "Codex for OSS için star rica" deme — araştırmamıza göre işe yaramıyor
- ❌ Başlıkta "amaçım X star toplamak" gibi şeyler yazma
- ❌ Modlar tarafından kaldırılırsa, hemen tekrar paylaşma — önce kuralları oku
