"""Clinical Metric Cards and Summary Components."""

from typing import Dict, Optional
import streamlit as st


def render_overview_cards():
    """Render the 4 key metric tiles for the Tổng quan dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.html("""
            <div class="metric-tile">
                <div class="metric-tile-label">Mô hình sàng lọc</div>
                <div class="metric-tile-value" style="font-size: 1.2rem;">Swin V2-Tiny</div>
                <div class="metric-tile-sub">27,57 triệu tham số • Sàng lọc</div>
            </div>
            """)

    with col2:
        st.html("""
            <div class="metric-tile">
                <div class="metric-tile-label">Mô hình phân vùng</div>
                <div class="metric-tile-value" style="font-size: 1.2rem;">Swin Hybrid</div>
                <div class="metric-tile-sub">38,80 triệu tham số • nnU-Net CNN</div>
            </div>
            """)

    with col3:
        st.html("""
            <div class="metric-tile">
                <div class="metric-tile-label">Dữ liệu đầu vào</div>
                <div class="metric-tile-value" style="font-size: 1.2rem;">Ảnh MRI não</div>
                <div class="metric-tile-sub">Lát cắt 2D • Kích thước gốc</div>
            </div>
            """)

    with col4:
        st.html("""
            <div class="metric-tile">
                <div class="metric-tile-label">Quy trình phân tích</div>
                <div class="metric-tile-value" style="font-size: 1.2rem;">AI hai giai đoạn</div>
                <div class="metric-tile-sub">Sàng lọc và phân vùng</div>
            </div>
            """)


def render_clinical_summary_card(
    screening_result: str,
    probability: float,
    segmentation_status: str,
    detected_regions: str,
    tumor_area: int,
    tumor_ratio: float = 0.0,
):
    """Render the hospital-grade AI Analysis Summary Sheet."""
    st.html(f"""
        <div class="clinical-summary-sheet">
            <div class="clinical-summary-title">
                <span style="display:inline-block; width:10px; height:10px; background:#0284C7; border-radius: 20px;"></span>
                PHIẾU TÓM TẮT KẾT QUẢ PHÂN TÍCH
            </div>
            <div class="clinical-summary-row">
                <span class="summary-key">Kết quả sàng lọc</span>
                <span class="summary-val" style="color: {'#BE123C' if 'Dương tính' in screening_result else '#059669'};">
                    {screening_result}
                </span>
            </div>
            <div class="clinical-summary-row">
                <span class="summary-key">Điểm sàng lọc</span>
                <span class="summary-val">{probability:.1f}%</span>
            </div>
            <div class="clinical-summary-row">
                <span class="summary-key">Trạng thái phân vùng</span>
                <span class="summary-val">{segmentation_status}</span>
            </div>
            <div class="clinical-summary-row">
                <span class="summary-key">Số vùng nghi ngờ</span>
                <span class="summary-val">{detected_regions}</span>
            </div>
            <div class="clinical-summary-row">
                <span class="summary-key">Diện tích vùng dự đoán</span>
                <span class="summary-val">{tumor_area:,} px ({tumor_ratio:.2f}%)</span>
            </div>
        </div>
        """)


def render_medical_disclaimer():
    """Render the mandatory clinical disclaimer at the bottom of pages."""
    st.html("""
        <div class="medical-disclaimer">
            <strong>Chỉ sử dụng cho nghiên cứu</strong> — Hệ thống phục vụ nghiên cứu và đào tạo; không thay thế việc đọc phim, chẩn đoán hoặc đánh giá chuyên môn của bác sĩ.
        </div>
        """)
