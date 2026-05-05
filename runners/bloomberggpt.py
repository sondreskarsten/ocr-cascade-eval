from shared import for_each_pdf, run_with_metrics


def main():
    def per_pdf(pdf_id, b):
        return {"note": "BloombergGPT is closed source — no public checkpoint. Documenting for completeness.",
                "pdf_chars": len(b["full_text"])}

    return {"status": "documented_only",
            "checkpoint": "BloombergGPT (Bloomberg L.P., closed)",
            "paper": "https://arxiv.org/abs/2303.17564",
            "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("bloomberggpt", main)
