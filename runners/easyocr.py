from shared import for_each_pdf, run_with_metrics


def main():
    import easyocr

    reader = easyocr.Reader(["no", "en"], gpu=False, verbose=False)

    def per_pdf(pdf_id, b):
        pages = []
        for img_path in b["page_imgs"]:
            res = reader.readtext(img_path)
            text = "\n".join(t[1] for t in res)
            confs = [t[2] for t in res]
            pages.append({"page_n": int(img_path.split("/")[-1].split("-")[1].split(".")[0]),
                          "n_lines": len(res),
                          "avg_conf": round(sum(confs) / max(len(confs), 1), 3),
                          "text": text})
        return {"n_pages": len(pages),
                "total_chars": sum(len(p["text"]) for p in pages),
                "pages": pages}

    return {"engine": "EasyOCR (Norwegian+English)", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("easyocr", main)
