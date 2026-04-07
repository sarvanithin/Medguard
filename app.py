"""
MedGuard — Health Universe deployment
Embeds the live Render UI directly so the look and feel is identical.
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MedGuard — Clinical LLM Safety",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }
        header[data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe(
    "https://medguard-ru03.onrender.com",
    height=900,
    scrolling=True,
)
