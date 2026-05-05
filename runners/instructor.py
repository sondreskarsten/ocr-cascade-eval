from shared import fetch_fixture, run_with_metrics


def main():
    fx = fetch_fixture()
    import json
    samples = json.loads(open(fx["samples.json"]).read())

    info = {"library": "instructor"}
    try:
        import instructor
        info["version"] = getattr(instructor, "__version__", "?")
    except Exception as e:
        info["import_error"] = f"{type(e).__name__}: {e}"
        return info

    from typing import Literal
    from pydantic import BaseModel

    canonicals = samples["canonical_titles"][:30]
    LiteralType = Literal[tuple(canonicals)]

    class Mapping(BaseModel):
        label: str
        canonical: LiteralType
        confidence: float

    info["pydantic_schema"] = Mapping.model_json_schema()
    info["constraint_demo"] = {
        "input_label": samples["norwegian_label"],
        "constrained_to": canonicals,
        "n_choices": len(canonicals),
        "note": "instructor wraps an LLM client (OpenAI/Anthropic/Gemini). No API key in this run; smoke test only."
    }
    return info


if __name__ == "__main__":
    run_with_metrics("instructor", main)
