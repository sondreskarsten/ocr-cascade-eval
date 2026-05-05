from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"engine": "marker"}
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(fx["test.pdf"])
        text, _, images = text_from_rendered(rendered)
        info["n_chars"] = len(text)
        info["text_head"] = text[:2000]
        info["n_images"] = len(images) if images else 0
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("marker", main)
