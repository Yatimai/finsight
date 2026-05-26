"""
LLM-as-judge answer-accuracy (offline grading of a saved eval run).

For each ground-truth question, compares the saved generated_answer to the gold
expected_answer via Claude Haiku, focusing on the key facts/figures and ignoring
phrasing, cited-page, or secondary differences. Abstention questions are graded
in code (no API call).

Usage: .venv/bin/python -m evaluation.llm_judge <result_file.json> [--model X]
Cost: ~50 Haiku calls (a few cents).
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

ENV_PATH = os.environ.get("FINSIGHT_ENV_FILE", "API_KEYS.env")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM = """Tu es un évaluateur financier rigoureux. On te donne une QUESTION sur un rapport annuel, une RÉPONSE DE RÉFÉRENCE (gold, extraite à la main du document, fiable), et une RÉPONSE GÉNÉRÉE par un système RAG.

Détermine si la réponse générée est factuellement correcte PAR RAPPORT AU GOLD. Concentre-toi sur les CHIFFRES CLÉS et les faits principaux. Ignore : les différences de formulation, les numéros de page cités, les détails secondaires, les explications en plus.

Normalise les chiffres avant de juger (FR : espace = séparateur milliers, virgule = décimale ; "84 683 M€" = 84683 millions).

Verdicts possibles :
- CORRECT : le ou les chiffres/faits clés sont justes (tolérance d'arrondi).
- PARTIAL : partiellement juste (un chiffre clé juste mais un autre faux/manquant, OU une erreur secondaire comme une date erronée).
- WRONG : chiffre clé faux, ou réponse hors sujet.
- ABSTAINED : le système a refusé de répondre ("n'apparaît pas", "ne figure pas").

Réponds UNIQUEMENT en JSON : {"verdict": "CORRECT|PARTIAL|WRONG|ABSTAINED", "reason": "<1 phrase>"}"""

ABSTAIN_KW = ["n'apparaît pas", "ne figure pas", "pas dans les documents", "pas disponible dans"]


def load_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    for p in (ENV_PATH,):
        try:
            for line in Path(p).read_text().splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    raise RuntimeError("ANTHROPIC_API_KEY introuvable")


def is_abstention(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ABSTAIN_KW)


def items(d):
    if isinstance(d, list):
        return d
    for k in ("questions", "results"):
        if isinstance(d, dict) and k in d:
            return d[k]
    return list(d.values())


def judge_one(client, model, question, gold, gen):
    user = f"QUESTION:\n{question}\n\nRÉPONSE DE RÉFÉRENCE (gold):\n{gold}\n\nRÉPONSE GÉNÉRÉE:\n{gen}"
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.0,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            txt = resp.content[0].text.strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            data = json.loads(m.group(0) if m else txt)
            return data.get("verdict", "?").upper(), data.get("reason", "")
        except Exception as e:
            if attempt == 3:
                return "ERROR", str(e)[:80]
            time.sleep(1.5 * (attempt + 1))


def main():
    res_path = sys.argv[1]
    model = DEFAULT_MODEL
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    client = anthropic.Anthropic(api_key=load_key())
    gt = {q["id"]: q for q in items(json.loads(Path("evaluation/ground_truth.json").read_text()))}
    res = {r["question_id"]: r for r in items(json.loads(Path(res_path).read_text()))}

    rows = []
    cat_tot = defaultdict(int)
    cat_ok = defaultdict(int)
    cat_partial = defaultdict(int)

    for qid, g in gt.items():
        cat = g.get("category", "?")
        gold = (g.get("expected_answer") or "").strip()
        gen = (res.get(qid, {}) or {}).get("generated_answer", "") or ""
        cat_tot[cat] += 1

        if cat == "abstention" or not gold:
            verdict = "CORRECT" if is_abstention(gen) else "WRONG"
            reason = "abstention attendue " + ("respectée" if verdict == "CORRECT" else "NON respectée")
        else:
            verdict, reason = judge_one(client, model, g["question"], gold, gen)

        if verdict == "CORRECT":
            cat_ok[cat] += 1
        elif verdict == "PARTIAL":
            cat_partial[cat] += 1
        rows.append((qid, cat, verdict, reason))
        print(f"{qid:>4} [{cat:<12}] {verdict:<9} {reason[:90]}")

    total = sum(cat_tot.values())
    okc = sum(cat_ok.values())
    partial = sum(cat_partial.values())

    print("\n--- Par catégorie (CORRECT / total, [PARTIAL]) ---")
    for cat in cat_tot:
        p = f" [+{cat_partial[cat]} partial]" if cat_partial[cat] else ""
        print(f"{cat:<14} {cat_ok[cat]:>2}/{cat_tot[cat]:<2} ({cat_ok[cat] / cat_tot[cat]:.0%}){p}")

    print(f"\n>>> ANSWER-ACCURACY (LLM-judge, {model})")
    print(f"    CORRECT strict : {okc}/{total} = {okc / total:.1%}")
    print(f"    CORRECT+PARTIAL : {okc + partial}/{total} = {(okc + partial) / total:.1%}")


if __name__ == "__main__":
    main()
