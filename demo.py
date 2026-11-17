"""Offline Streamlit entry point for the fictional price example."""

import streamlit as st

from pysharpe.app.offline_example import render_offline_example


def main() -> None:
    """Render the example without starting the research dashboard."""

    st.set_page_config(page_title="PySharpe · Offline example", layout="centered")
    st.title("PySharpe")
    st.caption("A five-minute introduction · fictional data · no market-data requests")
    render_offline_example()


if __name__ == "__main__":
    main()
