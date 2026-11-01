"""Landing page for AI Real-Time Gym Coach V2.

Rendered inside the existing Streamlit app (main.py) on the same port.
It does not replace login, vision, coaching, or persistence code.
"""

import os

import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CSS_PATH = os.path.join(_ROOT, "static", "home_page.css")
_HTML_PATH = os.path.join(_ROOT, "static", "home_page.html")


def _read_file(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def render_home_page():
    """Render the landing page.

    Returns True when the visitor asks to open the existing login page
    via a Login / Get Started control.
    """
    if st.query_params.get("enter") == "login" or st.session_state.get("from_home_page"):
        st.session_state["from_home_page"] = True
        if "enter" in st.query_params:
            del st.query_params["enter"]
        return True

    css = _read_file(_CSS_PATH)
    html = _read_file(_HTML_PATH)
    st.markdown(f"<style>{css}</style>{html}", unsafe_allow_html=True)
    return False


if __name__ == "__main__":
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-Time Gym Coach V2",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.warning(
        "The landing page is served by the existing app. "
        "Start it with: streamlit run main.py --server.port 8505"
    )
    st.stop()
