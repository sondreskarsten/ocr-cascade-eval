from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    from PIL import Image
    pages = {}
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except Exception as e:
        return {"status": "error", "engine": "surya",
                "import_error": f"{type(e).__name__}: {e}"}
    rec = RecognitionPredictor()
    det = DetectionPredictor()
    for label, key in [("p02", "pages_p02.png"), ("p06", "pages_p06.png")]:
        img = Image.open(fx[key]).convert("RGB")
        preds = rec([img], [["no", "en"]], det)
        out = preds[0]
        lines = [tl.text for tl in out.text_lines]
        pages[label] = {"n_lines": len(lines), "text": "\n".join(lines)}
    return {"engine": "surya", "pages": pages}


if __name__ == "__main__":
    run_with_metrics("surya", main)
