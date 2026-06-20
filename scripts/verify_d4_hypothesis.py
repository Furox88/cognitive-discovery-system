"""D4 — cds.hypothesis kalitesi derinlemesine doğrulama.

Generator ve evaluator modüllerinin:
  - Output çeşitliliği (n hipotez → n farklı?)
  - Duplicate tespiti (farklı sorularda farklı ID/stmt?)
  - Edge-case: n=0, n=1, n çok büyük, boş research_question
  - Confidence dağılımı (0-1 arası, monotonic artış)
  - Domain yanlış/unknown → fallback
  - Determinizm (uuid harici)
"""

from __future__ import annotations

import sys
from collections import Counter

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis.generator import (
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)

FAIL = "\033[91m"
PASS = "\033[92m"
WARN = "\033[93m"
END = "\033[0m"

failures: list[str] = []
warnings: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = f"{PASS}PASS{END}" if cond else f"{FAIL}FAIL{END}"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name + (f": {detail}" if detail else ""))


def warn(name: str, detail: str) -> None:
    print(f"  [{WARN}WARN{END}] {name} — {detail}")
    warnings.append(name + f": {detail}")


# ---------------------------------------------------------------------------
# D4.1 — Output çeşitliliği & duplicate
# ---------------------------------------------------------------------------


def test_output_diversity() -> None:
    print("\n=== D4.1 Output çeşitliliği & duplicate ===")

    gen = SimpleOfflineGenerator()

    # n=3 farklı hipotez — statement'lar benzersiz olmalı
    hs = gen.generate("test question", Domain.COSMOLOGY, n=3)
    check("n=3 returns 3 hypotheses", len(hs) == 3, f"got {len(hs)}")
    stmts = [h.statement for h in hs]
    check("n=3 statements unique", len(set(stmts)) == 3, f"unique={len(set(stmts))}")
    ids = [h.id for h in hs]
    check("n=3 IDs unique", len(set(ids)) == 3)

    # n=2 ile farklı bir çağrı — farklı uuid
    hs2 = gen.generate("test question", Domain.COSMOLOGY, n=2)
    ids2 = [h.id for h in hs2]
    check("different call → different IDs", set(ids).isdisjoint(set(ids2)))

    # Duplicate statement across different research questions?
    # Aynı domain, farklı soru → statement domain'e bağlı olduğu için AYNI olabilir
    # ama research_question farklı olmalı
    ha = gen.generate("question A", Domain.COSMOLOGY, n=1)
    hb = gen.generate("question B", Domain.COSMOLOGY, n=1)
    check(
        "different research_question preserved",
        ha[0].research_question != hb[0].research_question,
        f"rq_a={ha[0].research_question!r}",
    )

    # n=1
    hs = gen.generate("q", Domain.PHYSICS, n=1)
    check("n=1 returns 1", len(hs) == 1)

    # n=0
    hs = gen.generate("q", Domain.PHYSICS, n=0)
    check("n=0 returns empty list", hs == [])

    # n > available templates: fallback generic hypotheses
    hs = gen.generate("custom q", Domain.PHYSICS, n=10)
    check("n=10 returns 10 (with fallback)", len(hs) == 10, f"got {len(hs)}")
    # Generic fallback hep aynı olmalı (question'a bağlı)
    fallback_stmts = [h.statement for h in hs[2:]]
    if len(set(fallback_stmts)) == 1:
        warn(
            "fallback hypotheses are duplicates",
            f"{len(fallback_stmts)} hipotez aynı statement'ı paylaşiyor: {fallback_stmts[0][:60]!r}",
        )

    # Empty research question
    hs = gen.generate("", Domain.COSMOLOGY, n=1)
    check("empty research_question accepted", len(hs) == 1)
    check("empty rq propagates", hs[0].research_question == "")


# ---------------------------------------------------------------------------
# D4.2 — Confidence dağılımı
# ---------------------------------------------------------------------------


def test_confidence_distribution() -> None:
    print("\n=== D4.2 Confidence dağılımı ===")

    gen = SimpleOfflineGenerator()
    hs = gen.generate("q", Domain.PHYSICS, n=5)

    confs = [h.confidence for h in hs]
    # Generator formülü: min(0.9, 0.4 + i*0.05) → [0.4, 0.45, 0.5, 0.55, 0.6]
    # (D4a düzeltmesi öncesi 0.45+i*0.05 idi ve n>=12'de 1.0'ı geçip
    # pydantic ValidationError fırlatıyordu; şimdi 0.9'da clamp'leniyor.)
    check("all confidences in [0,1]", all(0.0 <= c <= 1.0 for c in confs), f"confs={confs}")
    check(
        "confidences are strictly increasing",
        all(confs[i + 1] > confs[i] for i in range(len(confs) - 1)),
        f"confs={confs}",
    )
    # İlk confidence 0.4 olmalı (0.4 + 0*0.05)
    check("first confidence == 0.4", abs(confs[0] - 0.4) < 1e-9, f"first={confs[0]}")

    # Çok büyük n: confidence 0.9 clamp'inde kalmalı.
    # D4a öncesi: 0.45+i*0.05 formülü n>=12'de 1.0'ı geçip ValidationError
    # fırlatıyordu. D4a düzeltmesi: min(0.9, 0.4+i*0.05). Bu test artık
    # clamp davranışını doğruluyor — eski "overflow bug" tespiti değil.
    try:
        hs_big = gen.generate("q", Domain.PHYSICS, n=20)
        confs_big = [h.confidence for h in hs_big]
        check(
            "confidence stays <= 0.9 even for large n (clamp works)",
            max(confs_big) <= 0.9 + 1e-9,
            f"max={max(confs_big)}",
        )
    except Exception as e:
        check(
            "large n=20 generates without error",
            False,
            f"{type(e).__name__}: {str(e)[:120]}",
        )


# ---------------------------------------------------------------------------
# D4.3 — Domain fallback
# ---------------------------------------------------------------------------


def test_domain_handling() -> None:
    print("\n=== D4.3 Domain handling & fallback ===")

    gen = SimpleOfflineGenerator()

    # String domain
    hs = gen.generate("q", "cosmology", n=2)
    check("string domain 'cosmology' → Domain.COSMOLOGY", hs[0].domain == Domain.COSMOLOGY)

    # Case-insensitive
    hs = gen.generate("q", "COSMOLOGY", n=1)
    check("case-insensitive domain mapping", hs[0].domain == Domain.COSMOLOGY)

    # Unknown domain string → GENERAL_SCIENCE fallback (also no templates → uses PHYSICS fallback list internally)
    hs = gen.generate("q", "nonexistent_domain", n=2)
    # Kod: .get(domain, templates[PHYSICS]) → PHYSICS listesini kullanır ama domain alanı GENERAL_SCIENCE kalır
    check(
        "unknown domain → domain field set to GENERAL_SCIENCE",
        hs[0].domain == Domain.GENERAL_SCIENCE,
        f"domain={hs[0].domain}",
    )

    # Domain enum doğrudan
    hs = gen.generate("q", Domain.MATHEMATICS, n=1)
    check("enum domain works", hs[0].domain == Domain.MATHEMATICS)

    # Tüm desteklenen domain'ler için smoke test
    for d in Domain:
        try:
            hs = gen.generate("q", d, n=1)
            check(f"domain {d.value!r} generates without error", len(hs) == 1)
        except Exception as e:
            check(f"domain {d.value!r} generates without error", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# D4.4 — Hypothesis model doğruluğu
# ---------------------------------------------------------------------------


def test_hypothesis_model() -> None:
    print("\n=== D4.4 Hypothesis model doğruluğu ===")

    gen = SimpleOfflineGenerator()
    hs = gen.generate("test rq", Domain.COSMOLOGY, n=3)

    for i, h in enumerate(hs, 1):
        check(f"h{i} has non-empty statement", bool(h.statement))
        check(f"h{i} has non-empty rationale", bool(h.rationale))
        check(f"h{i} has at least 1 assumption", len(h.assumptions) >= 1)
        check(f"h{i} has at least 1 prediction", len(h.predictions) >= 1)
        check(f"h{i} has valid status", h.status == HypothesisStatus.NEW)
        check(f"h{i} has tags", len(h.tags) >= 1)
        check(f"h{i} ID format H-xxxxxxxx", h.id.startswith("H-") and len(h.id) == 10)

    # research_question propagate
    check("research_question propagates", hs[0].research_question == "test rq")

    # tags domain içeriyor
    check("tags include domain value", "cosmology" in hs[0].tags or Domain.COSMOLOGY.value in hs[0].tags)


# ---------------------------------------------------------------------------
# D4.5 — PromptTemplate
# ---------------------------------------------------------------------------


def test_prompt_template() -> None:
    print("\n=== D4.5 PromptTemplate ===")

    p = PromptTemplate.render("what is X?", Domain.COSMOLOGY, n=5)
    check("render returns non-empty", bool(p))
    check("render contains research_question", "what is X?" in p)
    check("render contains domain value", "cosmology" in p)
    check("render contains n", "5 distinct hypotheses" in p)
    # Format string placeholders dolduruldu mu?
    check("no unfilled {placeholder}", "{" not in p and "}" not in p)

    # SYSTEM prompt statik ve hazır
    check("SYSTEM is non-empty string", isinstance(PromptTemplate.SYSTEM, str) and len(PromptTemplate.SYSTEM) > 20)


# ---------------------------------------------------------------------------
# D4.6 — generate_hypotheses convenience function
# ---------------------------------------------------------------------------


def test_convenience_function() -> None:
    print("\n=== D4.6 generate_hypotheses convenience ===")

    # Default generator
    hs = generate_hypotheses("q", Domain.PHYSICS, n=2)
    check("default generator works", len(hs) == 2)

    # Custom generator
    class CustomGen:
        def generate(self, research_question, domain=Domain.GENERAL_SCIENCE, n=3, **kwargs):
            return [
                Hypothesis(
                    id=f"CUSTOM-{i}",
                    statement=f"custom stmt {i}",
                    domain=Domain.GENERAL_SCIENCE,
                    research_question=research_question,
                    rationale="r",
                    assumptions=["a"],
                    predictions=["p"],
                    status=HypothesisStatus.NEW,
                    confidence=0.5,
                    tags=["custom"],
                )
                for i in range(n)
            ]

    hs = generate_hypotheses("q", n=2, generator=CustomGen())
    check("custom generator injected", len(hs) == 2 and hs[0].id == "CUSTOM-0")


# ---------------------------------------------------------------------------
# D4.7 — Determinizm (uuid harici alanlar)
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    print("\n=== D4.7 Determinizm (uuid harici) ===")

    gen = SimpleOfflineGenerator()
    # Aynı domain ve soru → statement'lar ve yapı aynı olmalı (uuid hariç)
    hs1 = gen.generate("same q", Domain.COSMOLOGY, n=3)
    hs2 = gen.generate("same q", Domain.COSMOLOGY, n=3)

    stmts1 = [h.statement for h in hs1]
    stmts2 = [h.statement for h in hs2]
    check(
        "statements deterministic across calls",
        stmts1 == stmts2,
        f"stmts1={stmts1[:1]}\nstmts2={stmts2[:1]}",
    )

    assumptions1 = [h.assumptions for h in hs1]
    assumptions2 = [h.assumptions for h in hs2]
    check("assumptions deterministic", assumptions1 == assumptions2)

    # Sadece uuid farklı
    ids1 = {h.id for h in hs1}
    ids2 = {h.id for h in hs2}
    check("IDs differ (uuid)", ids1.isdisjoint(ids2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    test_output_diversity()
    test_confidence_distribution()
    test_domain_handling()
    test_hypothesis_model()
    test_prompt_template()
    test_convenience_function()
    test_determinism()

    print("\n" + "=" * 70)
    print(f"Toplam: {len(failures)} hata, {len(warnings)} uyarı")
    if failures:
        print(f"\n{FAIL}HATALAR:{END}")
        for f in failures:
            print(f"  - {f}")
    if warnings:
        print(f"\n{WARN}UYARILAR:{END}")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
