from shared import for_each_pdf, run_with_metrics


def main():
    import pandas as pd
    import pandera.pandas as pa
    import re

    schema = pa.DataFrameSchema({
        "page_n": pa.Column(int, checks=pa.Check.greater_than(0)),
        "n_chars": pa.Column(int, checks=pa.Check.ge(0)),
        "n_norwegian_chars": pa.Column(int, checks=pa.Check.ge(0)),
    })
    nordics = set("æøåÆØÅ")

    def per_pdf(pdf_id, b):
        rows = []
        for k, txt in b["page_text"].items():
            n_nor = sum(1 for ch in txt if ch in nordics)
            rows.append({"page_n": int(k), "n_chars": len(txt), "n_norwegian_chars": n_nor})
        df = pd.DataFrame(rows)
        try:
            schema.validate(df, lazy=True)
            valid = True
            errs = []
        except Exception as e:
            valid = False
            errs = [str(e)[:300]]
        return {"validated": valid, "errors": errs,
                "summary": {"n_pages": len(df), "total_chars": int(df["n_chars"].sum()),
                            "total_norwegian_chars": int(df["n_norwegian_chars"].sum()),
                            "norwegian_ratio": round(df["n_norwegian_chars"].sum()/max(df["n_chars"].sum(),1), 4)}}

    return {"library": "pandera", "per_pdf": for_each_pdf(per_pdf)}


if __name__ == "__main__":
    run_with_metrics("pandera", main)
