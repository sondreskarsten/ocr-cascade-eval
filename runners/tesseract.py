from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import pytesseract
    from PIL import Image

    pages = {}
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        img = Image.open(fx[key])
        text = pytesseract.image_to_string(img, lang="nor")
        data = pytesseract.image_to_data(img, lang="nor", output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if int(c) > 0]
        pages[label] = {
            "n_lines": len(text.splitlines()),
            "n_words": len(confs),
            "avg_conf": round(sum(confs) / max(len(confs), 1), 1),
            "text": text,
        }
    return {"pages": pages, "engine": "tesseract", "lang": "nor"}


if __name__ == "__main__":
    run_with_metrics("tesseract", main)
