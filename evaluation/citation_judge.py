"""
Citation faithfulness judge (offline, multimodal).

For each [Page X] citation in a saved eval run's generated answers, renders the
cited PDF page as an image and asks Claude (Sonnet 4.6) whether that page
actually supports the claim(s) attributed to [Page X] in the answer.

This measures genuine citation faithfulness (does the cited evidence support the
claim) and is distinct from:
  - citation_accuracy in evaluate.py (mechanical: cited page in retrieved set)
  - answer accuracy in llm_judge.py (Haiku, text-only, answer vs gold)

The judge model defaults to Sonnet 4.6 to match the runtime verifier, so the
resulting number genuinely corresponds to "Sonnet 4.6 verification".

NOTE: the judge sees pages downscaled to ~1568px (the Sonnet image cap), while
Opus 4.7 generated at 2576px. On dense tables the judge may fail to read a
figure that is genuinely present, so this rate is a LOWER BOUND.

Usage:
  .venv/bin/python -m evaluation.citation_judge <result.json> \
      [--model claude-sonnet-4-6] [--qids q01,q02,q04] [--limit N]
"""

import base64
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import fitz  # pymupdf

ENV_PATH = os.environ.get("FINSIGHT_ENV_FILE", "API_KEYS.env")
DEFAULT_MODEL = "claude-sonnet-4-6"
DOCS_DIR = "data/documents"
MAX_PX = 1568  # Sonnet/Haiku image cap; Opus 4.7 generated at 2576px (see module note)

JUDGE_SYSTEM = """Tu es un vérificateur de citations dans des rapports financiers. On te donne une QUESTION, une RÉPONSE générée par un système RAG (qui contient des marqueurs de citation [Page X]), et l'IMAGE d'une page citée.

Ta tâche : déterminer si CETTE PAGE soutient réellement l'affirmation associée à [Page X] dans la réponse. Concrètement : le ou les chiffres/faits que la réponse attribue à cette page apparaissent-ils bien sur l'image fournie ?

Normalise les nombres (FR : espace = séparateur de milliers, virgule = décimale ; "84 683 M€" = 84683 millions).

Verdicts :
- SUPPORTED : le(s) chiffre(s)/fait(s) cité(s) apparaissent clairement sur la page (tolérance d'arrondi).
- PARTIAL : la page traite du sujet et soutient une partie de l'affirmation, mais pas le chiffre exact cité (ou un seul de plusieurs).
- NOT_SUPPORTED : le chiffre/fait cité n'est pas sur cette page, ou la page est sans rapport.

IMPORTANT — pagination : le numéro imprimé en bas/coin de la page peut différer du numéro cité (décalage normal entre page physique du PDF et numéro imprimé : pages de garde, sommaire, etc.). Ne pénalise JAMAIS pour un écart de numéro de page. L'image fournie EST la bonne page. Juge UNIQUEMENT si son CONTENU soutient l'affirmation.

Juge UNIQUEMENT d'après l'image fournie. Si la page est illisible ou trop dense pour lire le chiffre, dis-le dans "evidence" et mets PARTIAL.

Réponds UNIQUEMENT en JSON, en COMMENÇANT DIRECTEMENT par l'accolade ouvrante, sans aucun raisonnement ni texte avant : {"verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED", "evidence": "<ce que tu vois sur la page, 1 phrase>"}"""


def load_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        for line in Path(ENV_PATH).read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    raise RuntimeError(f"ANTHROPIC_API_KEY introuvable (ni env, ni {ENV_PATH})")


def items(d):
    if isinstance(d, list):
        return d
    for k in ("questions", "results"):
        if isinstance(d, dict) and k in d:
            return d[k]
    return list(d.values())


def render_page(pdf_path: str, page_num: int, max_px: int = MAX_PX) -> bytes | None:
    """Render physical (1-indexed) PDF page to PNG bytes, long edge <= max_px."""
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            return None
        page = doc[page_num - 1]
        rect = page.rect
        zoom = max_px / max(rect.width, rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        png = pix.tobytes("png")
        doc.close()
        return png
    except Exception:
        return None


def judge_citation(client, model, question, answer, page_num, img_bytes):
    b64 = base64.standard_b64encode(img_bytes).decode()
    user = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        {
            "type": "text",
            "text": (
                f"QUESTION:\n{question}\n\nRÉPONSE GÉNÉRÉE (avec citations):\n{answer}\n\n"
                f"La réponse attribue une ou plusieurs affirmations à [Page {page_num}]. "
                f"L'image ci-dessus EST cette page (ignore tout numéro imprimé dessus, c'est un décalage normal). "
                f"Son contenu soutient-il ces affirmations ?"
            ),
        },
    ]
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=700,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            txt = next((b.text for b in resp.content if b.type == "text"), "").strip()
            import re

            m = re.search(r"\{.*\}", txt, re.DOTALL)
            data = json.loads(m.group(0) if m else txt)
            return data.get("verdict", "?").upper(), data.get("evidence", "")
        except Exception as e:
            if attempt == 3:
                return "ERROR", str(e)[:80]
            time.sleep(1.5 * (attempt + 1))


def main():
    res_path = sys.argv[1]
    model = DEFAULT_MODEL
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    only_qids = None
    if "--qids" in sys.argv:
        only_qids = set(sys.argv[sys.argv.index("--qids") + 1].split(","))
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    client = anthropic.Anthropic(api_key=load_key())
    gt = {q["id"]: q for q in items(json.loads(Path("evaluation/ground_truth.json").read_text()))}
    res = {r["question_id"]: r for r in items(json.loads(Path(res_path).read_text()))}

    rows = []
    n_cit = defaultdict(int)  # SUPPORTED / PARTIAL / NOT_SUPPORTED / ERROR
    cat_tot = defaultdict(int)
    cat_sup = defaultdict(int)
    q_done = 0

    for qid, g in gt.items():
        if only_qids and qid not in only_qids:
            continue
        if g.get("category") == "abstention" or not g.get("source_pages"):
            continue
        r = res.get(qid)
        if not r:
            continue
        cited = r.get("cited_pages") or []
        if not cited:
            continue
        if limit is not None and q_done >= limit:
            break
        q_done += 1

        doc = g.get("source_document", "")
        pdf_path = os.path.join(DOCS_DIR, doc)
        cat = g.get("category", "?")
        answer = r.get("generated_answer", "") or ""

        for p in cited:
            img = render_page(pdf_path, p)
            if img is None:
                verdict, ev = "NOT_SUPPORTED", f"page {p} hors limites du PDF"
            else:
                verdict, ev = judge_citation(client, model, g["question"], answer, p, img)
            n_cit[verdict] += 1
            cat_tot[cat] += 1
            if verdict == "SUPPORTED":
                cat_sup[cat] += 1
            rows.append((qid, cat, p, verdict, ev))
            print(f"{qid:>4} [{cat:<12}] Page {p:<4} {verdict:<14} {ev[:80]}")

    total = sum(n_cit.values())
    sup = n_cit["SUPPORTED"]
    par = n_cit["PARTIAL"]
    if total == 0:
        print("\nAucune citation jugée.")
        return

    print("\n--- Par catégorie (SUPPORTED / total citations) ---")
    for cat in cat_tot:
        print(f"{cat:<14} {cat_sup[cat]:>2}/{cat_tot[cat]:<2} ({cat_sup[cat] / cat_tot[cat]:.0%})")

    print(f"\n>>> CITATION FAITHFULNESS (LLM-judge multimodal, {model})")
    print(f"    questions jugées       : {q_done}")
    print(f"    citations jugées       : {total}")
    print(f"    SUPPORTED              : {sup}/{total} = {sup / total:.1%}")
    print(f"    SUPPORTED+PARTIAL      : {sup + par}/{total} = {(sup + par) / total:.1%}")
    print(f"    breakdown              : {dict(n_cit)}")
    print("    (borne basse : juge à 1568px, generation à 2576px)")

    # Save
    stem = Path(res_path).stem
    out = Path("evaluation/results") / f"{stem}_citation_judged.json"
    out.write_text(
        json.dumps(
            {
                "model": model,
                "questions_judged": q_done,
                "citations_total": total,
                "supported": sup,
                "partial": par,
                "support_rate": sup / total,
                "support_or_partial_rate": (sup + par) / total,
                "breakdown": dict(n_cit),
                "rows": [{"qid": q, "category": c, "page": p, "verdict": v, "evidence": e} for (q, c, p, v, e) in rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n    -> {out}")


if __name__ == "__main__":
    main()
