from shared import fetch_fixture, run_with_metrics


def main():
    return {"library": "bloomberggpt",
            "status": "documented_only",
            "checkpoint": "Bloomberg/BloombergGPT (50.6B)",
            "license": "closed-source / proprietary",
            "release_paper": "arXiv:2303.17564",
            "note": "Cannot be loaded — weights not publicly distributed."}


if __name__ == "__main__":
    run_with_metrics("bloomberggpt", main)
