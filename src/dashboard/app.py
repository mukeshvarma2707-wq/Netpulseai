"""
src/dashboard/app.py

Streamlit dashboard for BalanceGrid. Talks to the FastAPI backend over
HTTP - same architecture as CounterFlag's dashboard.

REQUIREMENTS:
    pip install streamlit requests

RUN (two terminals, both must stay open):
    Terminal 1: uvicorn src.api.main:app --reload
    Terminal 2: streamlit run src/dashboard/app.py
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="BalanceGrid", layout="wide")
st.title("BalanceGrid - Network Congestion Diagnosis Dashboard")


def fetch_cases(classification: str, limit: int):
    try:
        params = {"limit": limit}
        if classification != "All":
            params["classification"] = classification
        resp = requests.get(f"{API_BASE}/cases", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def fetch_case_detail(case_id: int):
    resp = requests.get(f"{API_BASE}/cases/{case_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def trigger_report(case_id: int):
    resp = requests.post(f"{API_BASE}/cases/{case_id}/report", timeout=30)
    resp.raise_for_status()
    return resp.json()


with st.sidebar:
    st.header("Settings")
    classification_filter = st.selectbox("Filter by classification", ["All", "ROUTINE", "ANOMALOUS"])
    limit = st.slider("Number of cases to show", 5, 100, 20)
    if st.button("Refresh"):
        st.rerun()

cases, error = fetch_cases(classification_filter, limit)

if error:
    st.error(
        f"Couldn't reach the API at {API_BASE}. Make sure it's running: "
        f"`uvicorn src.api.main:app --reload`\n\nDetails: {error}"
    )
    st.stop()

if not cases:
    st.warning("No diagnosed cases found. Run the pipeline and populate_db.py first.")
    st.stop()

st.subheader(f"Showing {len(cases)} diagnosed cases")

table_data = [
    {
        "Cell": c["cell_id"],
        "Target Time": c["target_datetime"],
        "Classification": c["classification"],
        "Forecast": round(c["naive_forecast"], 1),
        "Threshold": round(c["congestion_threshold"], 1),
        "Deficit": round(c["deficit"], 1) if c["deficit"] is not None else None,
        "Covered": round(c["covered"], 1) if c["covered"] is not None else None,
        "Fully Resolved?": "Yes" if c["fully_resolved"] else ("No" if c["fully_resolved"] is not None else None),
        "Report?": "Yes" if c["has_report"] else "No",
    }
    for c in cases
]
st.dataframe(table_data, width="stretch")

case_options = {f"Cell {c['cell_id']} @ {c['target_datetime']} (id {c['id']})": c["id"] for c in cases}
selected_label = st.selectbox("Select a case for details", list(case_options.keys()))
selected_id = case_options[selected_label]

detail = fetch_case_detail(selected_id)

st.subheader(f"Case Detail - Cell {detail['cell_id']} @ {detail['target_datetime']}")

badge_color = "orange" if detail["classification"] == "ROUTINE" else "red"
st.markdown(f"**Classification:** :{badge_color}[{detail['classification']}]")
st.write(f"**Reason:** {detail['reason']}")

col1, col2 = st.columns(2)
col1.metric("Forecasted Load", f"{detail['naive_forecast']:.1f}")
col2.metric("Congestion Threshold", f"{detail['congestion_threshold']:.1f}")

if detail["classification"] == "ROUTINE":
    st.markdown("### Reallocation Plan")
    if detail["sources"]:
        for s in detail["sources"]:
            st.write(f"- Move **{s['amount_moved']:.2f}** units from Cell **{s['source_cell_id']}**")
        col3, col4 = st.columns(2)
        col3.metric("Deficit", f"{detail['deficit']:.1f}" if detail["deficit"] is not None else "N/A")
        col4.metric("Covered", f"{detail['covered']:.1f}" if detail["covered"] is not None else "N/A")
        if detail["fully_resolved"]:
            st.success("Fully resolved by reallocation.")
        else:
            st.warning("Only partially resolved - insufficient nearby spare capacity for full coverage.")
    else:
        st.info("No reallocation was possible for this case (no spare capacity found nearby).")
else:
    st.warning("This case was classified as ANOMALOUS - no automatic reallocation was attempted. Recommend manual review.")

st.markdown("### Report")
if detail["report"] is None:
    st.info("No report generated yet for this case.")
    if st.button("Generate report"):
        with st.spinner("Generating report..."):
            trigger_report(selected_id)
            st.rerun()
else:
    st.write(detail["report"]["report_text"])
    st.caption(f"Generated at: {detail['report']['generated_at']}")
    if st.button("Regenerate report"):
        with st.spinner("Regenerating..."):
            trigger_report(selected_id)
            st.rerun()