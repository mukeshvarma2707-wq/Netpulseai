"""Streamlit dashboard for BalanceGrid."""
import streamlit as st

st.set_page_config(page_title="BalanceGrid", page_icon="⚡", layout="wide")
st.title("BalanceGrid")
st.caption("Grid demand forecasting, diagnosis, and dispatch planning")
st.info("Add processed Milano CDR data under data/processed/ to begin exploring forecasts.")
