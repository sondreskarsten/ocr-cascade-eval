"""tesseract TSV (word-level) — secondary numeric signal.

Why this exists: tesseract's `image_to_string` text mode applies line-merging
heuristics that drop value columns on Norwegian balanse-style layouts. The TSV
stream (`image_to_data`) preserves token-level bbox + confidence and lets us
re-cluster numbers correctly.

Empirical: on v2 fixture, recovers 22/24 truth values that tesseract text-mode
missed — same OCR engine, different post-processing.
"""
from shared import for_each_pdf, run_with_metrics
import re


def _tesseract_tsv_per_page(img_path, lang="nor"):
    import pytesseract
    data = pytesseract.image_to_data(
        img_path, lang=lang, output_type=pytesseract.Output.DICT)

    by_line = {}
    for i, w in enumerate(data["text"]):
        w = (w or "").strip()
        if not w: continue
        try: conf = float(data["conf"][i])
        except: conf = -1
        if conf < 30: continue
        line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        by_line.setdefault(line_id, []).append({
            "text": w,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "right": int(data["left"][i]) + int(data["width"][i]),
            "conf": conf,
        })

    found = []
    for line_id, tokens in by_line.items():
        tokens.sort(key=lambda t: t["left"])
        i = 0
        while i < len(tokens):
            t = tokens[i]["text"]
            if not re.match(r"^-?\d+$", t):
                i += 1
                continue
            sign = "-" if t.startswith("-") else ""
            digits = re.sub(r"[^\d]", "", t)
            j = i + 1
            x_anchor = tokens[i]["right"]
            within_gaps = []
            while j < len(tokens):
                tn = tokens[j]["text"]
                if not re.match(r"^\d{1,3}$", tn):
                    break
                gap = tokens[j]["left"] - x_anchor
                if gap >= 130:
                    break
                if within_gaps:
                    median_gap = sorted(within_gaps)[len(within_gaps)//2]
                    if gap > 2.5 * max(median_gap, 20) and gap > 50:
                        break
                within_gaps.append(gap)
                digits += tn
                x_anchor = tokens[j]["right"]
                j += 1
            try:
                v = int(sign + digits)
                if abs(v) >= 10:
                    avg_conf = sum(tokens[k]["conf"] for k in range(i, j)) / max(1, j - i)
                    found.append({
                        "value": v,
                        "left": tokens[i]["left"],
                        "top": tokens[i]["top"],
                        "right": x_anchor,
                        "n_tokens": j - i,
                        "avg_conf": round(avg_conf, 2),
                    })
            except: pass
            i = max(j, i + 1)
    return found


def main():
    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            try:
                nums = _tesseract_tsv_per_page(img_path)
                pages.append({"img": img_path.split("/")[-1],
                              "n_numbers": len(nums),
                              "numbers": nums})
            except Exception as e:
                pages.append({"img": img_path.split("/")[-1],
                              "error": f"{type(e).__name__}: {str(e)[:120]}"})
        all_values = sorted({p_n["value"]
                             for p in pages
                             for p_n in p.get("numbers", [])})
        return {"n_pages": len(pages),
                "n_unique_numbers": len(all_values),
                "all_values": all_values,
                "pages": pages}

    return {"runner": "tesseract_tsv (word-level numeric extraction)",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tesseract_tsv", main)
