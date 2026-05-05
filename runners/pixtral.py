from shared import for_each_pdf, run_with_metrics


def main():
    info = {"library": "pixtral (Mistral)"}
    info["note"] = "Pixtral-12B needs >24GB GPU; on CPU it is impractical. Recording placeholder."

    def per_pdf(pdf_id, b):
        return {"n_pages": b["n_pages"], "pdf_chars": len(b["full_text"])}

    return {**info, "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("pixtral", main)
