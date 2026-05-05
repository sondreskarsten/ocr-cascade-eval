from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import easyocr
    pages = {}
    reader = easyocr.Reader(["no", "en"], gpu=False, verbose=False)
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        res = reader.readtext(fx[key], detail=1)
        texts = [r[1] for r in res]
        confs = [r[2] for r in res]
        pages[label] = {"n_lines": len(texts),
                        "avg_conf": round(sum(confs)/max(len(confs),1), 3),
                        "text": "\n".join(texts)}
    return {"engine": "easyocr", "langs": ["no","en"], "pages": pages}


if __name__ == "__main__":
    run_with_metrics("easyocr", main)
