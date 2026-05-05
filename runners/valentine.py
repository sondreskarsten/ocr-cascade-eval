from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json, pandas as pd
    samples = json.loads(open(fx["samples.json"]).read())
    info = {"library": "valentine"}
    try:
        from valentine import valentine_match
        from valentine.algorithms import Coma
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info
    queries = [item["q"] for item in samples["queries"][:10]]
    cands = samples["canonical_titles"][:30]
    df_q = pd.DataFrame({c: [] for c in queries})
    df_c = pd.DataFrame({c: [] for c in cands})
    try:
        matcher = Coma()
        matches = valentine_match(df_q, df_c, matcher)
        info["matcher"] = "Coma"
        info["n_matches"] = len(matches)
        info["sample_matches"] = list(matches.items())[:10]
    except Exception as e:
        info["match_error"] = f"{type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    run_with_metrics("valentine", main)
