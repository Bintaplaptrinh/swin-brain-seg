"""Tổng quan Page View for Brain Tumor CADx."""

import streamlit as st

from app_components.cards import render_medical_disclaimer, render_overview_cards
from app_components.header import render_header
from app_components.pipeline_view import render_pipeline_diagram


def render_overview_page():
    """Render the Tổng quan Dashboard."""
    render_header(
        title="HỆ THỐNG PHÂN TÍCH MRI NÃO",
        subtitle="Hỗ trợ sàng lọc và xác định vùng nghi ngờ u não bằng trí tuệ nhân tạo",
    )

    # 4 Clinical Metric Cards
    render_overview_cards()

    # Visual Flowchart Section
    render_pipeline_diagram()

    st.info("Bắt đầu tại Phân tích MRI. Ảnh mô phỏng dùng để thử quy trình; suy luận thật cần tệp trọng số mô hình. Số liệu thí nghiệm cũ không xác nhận hiệu năng của ứng dụng hoặc ảnh hiện tại.")
    render_medical_disclaimer()
