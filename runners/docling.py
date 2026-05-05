from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    info = {"engine": "docling"}
    try:
        from docling.document_converter import DocumentConverter
        conv = DocumentConverter()
        result = conv.convert(fx["test.pdf"])
        md = result.document.export_to_markdown()
        info["n_chars"] = len(md)
        info["markdown_head"] = md[:2000]
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("docling", main)
