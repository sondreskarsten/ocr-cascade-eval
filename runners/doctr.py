from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    pages = {}
    pred = ocr_predictor(pretrained=True)
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        doc = DocumentFile.from_images(fx[key])
        res = pred(doc)
        out = res.export()
        lines = []
        for page in out["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    lines.append(" ".join(w["value"] for w in line["words"]))
        pages[label] = {"n_lines": len(lines), "text": "\n".join(lines)}
    return {"engine": "doctr", "pages": pages}


if __name__ == "__main__":
    run_with_metrics("doctr", main)
