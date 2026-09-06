"""
src/agents/report_agent.py

Wraps the diagnosis + solver output (already validated as correct in earlier
scripts) into natural-language reports for a human operator, using
LangGraph to route between two distinct report types.

WHY THIS COMES LAST, AFTER THE LOGIC IS VALIDATED: this agent explains and
frames results — it never decides congestion risk (that's the forecast +
threshold), never classifies routine vs. anomalous (that's the diagnosis
logic), and never computes reallocation amounts (that's the solver). All of
that was built and validated deterministically first. This script's only
job is turning already-correct structured results into readable text.

MOCK MODE: if no LLM credentials are configured (Azure OpenAI or OpenAI),
this falls back to a deterministic template instead of a real API call —
so you can validate the full pipeline's data flow and routing logic right
now, before wiring up real credentials. Set AZURE_OPENAI_ENDPOINT and
AZURE_OPENAI_KEY (or OPENAI_API_KEY) in your .env to use a real LLM.

REQUIREMENTS:
    pip install langgraph langchain-openai pandas pyarrow python-dotenv

INPUT:
    data/raw/diagnosis_results_1h.parquet (or .csv)
    data/raw/reallocation_results.parquet (or .csv)
    data/raw/reallocation_coverage.parquet (or .csv)

RUN:
    python src/agents/report_agent.py
"""

import os
from pathlib import Path
from typing import TypedDict

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class CaseState(TypedDict, total=False):
    CellID: int
    target_datetime: object
    classification: str
    reason: str
    naive_forecast_1h: float
    congestion_threshold: float
    deficit: float
    covered: float
    fully_resolved: bool
    reallocation_sources: list
    report: str


def load_case_data() -> pd.DataFrame:
    def _load(name):
        p_path, c_path = DATA_DIR / f"{name}.parquet", DATA_DIR / f"{name}.csv"
        if p_path.exists():
            return pd.read_parquet(p_path)
        if c_path.exists():
            return pd.read_csv(c_path)
        return None

    diagnosis = _load("diagnosis_results_1h")
    coverage = _load("reallocation_coverage")
    reallocation = _load("reallocation_results")

    if diagnosis is None:
        raise FileNotFoundError("Run src/diagnosis/diagnosis_agent.py first.")

    for col in ("target_datetime_1h",):
        if col in diagnosis.columns and not pd.api.types.is_datetime64_any_dtype(diagnosis[col]):
            diagnosis[col] = pd.to_datetime(diagnosis[col])

    if coverage is not None:
        if not pd.api.types.is_datetime64_any_dtype(coverage["target_datetime"]):
            coverage["target_datetime"] = pd.to_datetime(coverage["target_datetime"])
        diagnosis = diagnosis.merge(
            coverage.rename(columns={"target_datetime": "target_datetime_1h"}),
            on=["CellID", "target_datetime_1h"], how="left",
        )
    if reallocation is not None and not reallocation.empty:
        if not pd.api.types.is_datetime64_any_dtype(reallocation["target_datetime"]):
            reallocation["target_datetime"] = pd.to_datetime(reallocation["target_datetime"])
        reallocation = reallocation.rename(columns={"target_datetime": "target_datetime_1h", "congested_cell": "CellID"})

    return diagnosis, reallocation


def get_reallocation_sources(reallocation: pd.DataFrame, cell_id: int, target_dt) -> list:
    if reallocation is None or reallocation.empty:
        return []
    matches = reallocation[(reallocation["CellID"] == cell_id) & (reallocation["target_datetime_1h"] == target_dt)]
    return [(row["source_cell"], row["amount_moved"]) for _, row in matches.iterrows()]


def llm_available() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY")) or bool(
        os.environ.get("OPENAI_API_KEY")
    )


def get_llm():
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY"):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-05-01-preview",
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0.2,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ---- LangGraph nodes ----

def generate_routine_report(state: CaseState) -> CaseState:
    sources_text = ", ".join(f"{amt:.1f} units from Cell {src}" for src, amt in state.get("reallocation_sources", []))
    coverage_pct = (state["covered"] / state["deficit"] * 100) if state.get("deficit") else 0

    if llm_available():
        llm = get_llm()
        prompt = (
            f"Write a brief, plain-language operator note (2-3 sentences) for a routine, "
            f"explainable network congestion event. Cell {state['CellID']} is forecasted to "
            f"exceed its normal traffic threshold at {state['target_datetime']}, explained by: "
            f"{state['reason']}. The optimization solver recommends reallocating capacity: "
            f"{sources_text or 'no reallocation was possible'}, covering {coverage_pct:.0f}% of "
            f"the predicted deficit. State the facts plainly; do not invent numbers not given here."
        )
        report = llm.invoke(prompt).content
    else:
        report = (
            f"[MOCK] Cell {state['CellID']} is forecasted to exceed its normal traffic threshold "
            f"at {state['target_datetime']} ({state['reason']}). "
            f"Recommended action: {sources_text or 'no reallocation possible — insufficient nearby spare capacity'}. "
            f"This covers {coverage_pct:.0f}% of the predicted deficit. "
            f"Classified ROUTINE — safe for automatic handling, logged for audit."
        )
    state["report"] = report
    return state


def generate_anomalous_report(state: CaseState) -> CaseState:
    if llm_available():
        llm = get_llm()
        prompt = (
            f"Write a brief, plain-language escalation note (2-3 sentences) for a network "
            f"operator. Cell {state['CellID']} is forecasted to exceed its normal traffic "
            f"threshold at {state['target_datetime']}, but this is classified as ANOMALOUS "
            f"because: {state['reason']}. No automatic action was taken — recommend the "
            f"operator investigate. Do not invent facts not given here."
        )
        report = llm.invoke(prompt).content
    else:
        report = (
            f"[MOCK] Cell {state['CellID']} is forecasted to exceed its normal traffic threshold "
            f"at {state['target_datetime']}, but this does not match a routine, explainable "
            f"pattern ({state['reason']}). No automatic reallocation was attempted. "
            f"ESCALATED — recommend manual investigation before any action is taken."
        )
    state["report"] = report
    return state


def route(state: CaseState) -> str:
    return "routine" if state["classification"] == "ROUTINE" else "anomalous"


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(CaseState)
    graph.add_node("routine", generate_routine_report)
    graph.add_node("anomalous", generate_anomalous_report)
    graph.set_conditional_entry_point(route, {"routine": "routine", "anomalous": "anomalous"})
    graph.add_edge("routine", END)
    graph.add_edge("anomalous", END)
    return graph.compile()


if __name__ == "__main__":
    print(f"LLM credentials found: {llm_available()} "
          f"({'will call a real LLM' if llm_available() else 'running in MOCK mode — no API calls'})")

    diagnosis, reallocation = load_case_data()
    app = build_graph()

    sample = pd.concat([
        diagnosis[diagnosis["classification"] == "ROUTINE"].head(3),
        diagnosis[diagnosis["classification"] == "ANOMALOUS"].head(3),
    ])

    print(f"\nGenerating reports for {len(sample)} sample cases...\n")
    for _, row in sample.iterrows():
        sources = get_reallocation_sources(reallocation, row["CellID"], row["target_datetime_1h"])
        state: CaseState = {
            "CellID": row["CellID"],
            "target_datetime": row["target_datetime_1h"],
            "classification": row["classification"],
            "reason": row["reason"],
            "deficit": row.get("deficit", None),
            "covered": row.get("covered", 0),
            "reallocation_sources": sources,
        }
        result = app.invoke(state)
        print(f"--- Cell {row['CellID']} @ {row['target_datetime_1h']} [{row['classification']}] ---")
        print(result["report"])
        print()

    print("Done. Paste the printed output back so we can review the generated reports together.")