from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import pandas as pd
    import pandera.pandas as pa
    from pandera import Column, DataFrameSchema, Check

    df = pd.read_csv(fx["table.csv"])

    schema = DataFrameSchema({
        "label": Column(str, nullable=False),
        "value_2024": Column(int, nullable=True),
        "value_2023": Column(int, nullable=True),
    })

    sums_balance = df[df["label"].str.startswith("Sum")]
    bad = []
    try:
        schema.validate(df, lazy=True)
        validated = True
    except Exception as e:
        validated = False
        bad.append(str(e)[:600])

    sub_2024 = df.loc[df.label == "Annen driftskostnad", "value_2024"].sum()
    sum_2024 = df.loc[df.label == "Sum kostnader", "value_2024"].sum()

    return {"library": "pandera",
            "validated": validated,
            "errors": bad,
            "balance_check_2024": {
                "Annen driftskostnad": int(sub_2024),
                "Sum kostnader": int(sum_2024),
                "match": int(sub_2024) == int(sum_2024),
            }}


if __name__ == "__main__":
    run_with_metrics("pandera", main)
