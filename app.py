"""Brain Tumor CADx Platform - Main Streamlit Application Entrypoint.

Clinical-grade, modern healthcare AI dashboard for brain tumor screening and segmentation.
"""

from __future__ import annotations

import streamlit as st

# Configure page early
st.set_page_config(
    page_title="Phân tích MRI não | CADx",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Clinical Light Theme CSS
from app_utils.styles import CLINICAL_CSS
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

# Navigation & View imports
from app_components.sidebar import render_sidebar
from app_views.about_view import render_about_page
from app_views.analysis_view import render_analysis_page
from app_views.model_info_view import render_model_info_page
from app_views.overview_view import render_overview_page


def main():
    """Route user request to the designated clinical view."""
    active_page = render_sidebar()

    if active_page == "Tổng quan":
        render_overview_page()
    elif active_page == "Phân tích MRI":
        render_analysis_page()
    elif active_page == "Thông tin mô hình":
        render_model_info_page()
    elif active_page == "Giới thiệu":
        render_about_page()
    else:
        render_overview_page()


if __name__ == "__main__":
    main()
