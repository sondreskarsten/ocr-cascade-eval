from shared import for_each_pdf, run_with_metrics


def main():
    import pytesseract
    from PIL import Image

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang="nor")
            data = pytesseract.image_to_data(img, lang="nor", output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data["conf"] if str(c) not in ("", "-1") and int(c) > 0]
            pages.append({
                "page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                "n_words": len(confs),
                "avg_conf": round(sum(confs) / max(len(confs), 1), 1),
                "n_chars": len(text),
                "text": text,
            })
        return {"n_pages": len(pages),
                "total_chars": sum(p["n_chars"] for p in pages),
                "total_words": sum(p["n_words"] for p in pages),
                "pages": pages}

    return {"engine": "tesseract", "lang": "nor", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("tesseract", main)
