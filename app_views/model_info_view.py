"""Thông tin mô hình Page View for Brain Tumor CADx."""

import streamlit as st

from app_components.cards import render_medical_disclaimer
from app_components.header import render_header


def render_model_info_page():
    """Render technical specifications and model benchmark leaderboards."""
    render_header(
        title="Thông tin mô hình",
        subtitle="Kiến trúc, thành phần mạng và cấu hình mô hình của dự án.",
    )

    # 1. Tổng quan Comparison Cards
    col_m1, col_m2 = st.columns(2, gap="large")

    with col_m1:
        st.html("""
            <div class="med-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="font-size:1.1rem; font-weight:700; color:#0F2942;">Mô hình sàng lọc</div>
                    <span class="status-pill status-info">Giai đoạn 1</span>
                </div>
                <div style="font-size:1.2rem; font-weight:700; color:#0284C7; font-family:var(--app-font);">
                    Swin V2-Tiny
                </div>
                <div style="font-size: 1rem; color:#475569; margin:8px 0 16px 0;">
                    <strong>Chức năng:</strong> Sàng lọc từng lát cắt (không u / có u)
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius: 20px; padding:12px; font-size: 1rem; line-height:1.6; color:#334155;">
                    <div>• <strong>Mạng nền:</strong> <code>swinv2_cr_tiny_ns_224</code> (timm)</div>
                    <div>• <strong>Số tham số:</strong> 27,573,985 (~27.57M)</div>
                    <div>• <strong>Đầu vào:</strong> 224 × 224 pixels (Chuẩn hóa vùng khác 0)</div>
                    <div>• <strong>Đầu ra:</strong> Một logit &rarr; Sigmoid xác suất có u</div>
                    <div>• <strong>Ngưỡng sàng lọc:</strong> 0.40 (ngưỡng sàng lọc nghiên cứu)</div>
                </div>
            </div>
            """)

    with col_m2:
        st.html("""
            <div class="med-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="font-size:1.1rem; font-weight:700; color:#0F2942;">Mô hình phân vùng</div>
                    <span class="status-pill status-info">Giai đoạn 2</span>
                </div>
                <div style="font-size:1.2rem; font-weight:700; color:#07559B; font-family:var(--app-font);">
                    Swin Hybrid
                </div>
                <div style="font-size: 1rem; color:#475569; margin:8px 0 16px 0;">
                    <strong>Chức năng:</strong> Dự đoán vùng nghi ngờ u ở độ phân giải cao
                </div>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius: 20px; padding:12px; font-size: 1rem; line-height:1.6; color:#334155;">
                    <div>• <strong>Kiến trúc:</strong> nnU-Net Residual CNN + 2D Swin Transformer</div>
                    <div>• <strong>Số tham số:</strong> 38,801,022 (~38.80M có thể huấn luyện)</div>
                    <div>• <strong>Độ phức tạp:</strong> 237.78 GFLOPs (với đầu vào 512×512)</div>
                    <div>• <strong>Đầu ra:</strong> Mặt nạ nhị phân khôi phục theo kích thước ảnh gốc</div>
                </div>
            </div>
            """)

    # 2. Architecture Diagram Section
    st.markdown("#### Sơ đồ kiến trúc Swin Hybrid")
    st.html("""
        <p style="font-size: 1rem; color:#64748B; margin-bottom:18px;">
            Hợp nhất đặc trưng đa tỉ lệ giữa mạng tích chập và cơ chế tự chú ý theo cửa sổ.
        </p>
        <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius: 20px; padding:20px; text-align:center;">
            <div style="display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:10px; font-size: 1rem; font-weight:600;">
                <span style="background:#FFFFFF; border:1px solid #94A3B8; padding:8px 14px; border-radius: 20px; color:#0F2942;">
                    MRI thang xám (512×512)
                </span>
                <span style="color:#64748B;">&rarr;</span>
                <span style="background:#EFF6FF; border:1px solid #3B82F6; padding:8px 14px; border-radius: 20px; color:#1E40AF;">
                    Bộ mã hóa CNN phần dư (E1–E6)
                </span>
                <span style="color:#64748B;">+</span>
                <span style="background:#EDF4FA; border:1px solid #07559B; padding:8px 14px; border-radius: 20px; color:#064B87;">
                    Tháp Swin Transformer (24–384 kênh)
                </span>
                <span style="color:#64748B;">&rarr;</span>
                <span style="background:#F0FDF4; border:1px solid #10B981; padding:8px 14px; border-radius: 20px; color:#065F46;">
                    Hợp nhất phần dư đa tỉ lệ
                </span>
                <span style="color:#64748B;">&rarr;</span>
                <span style="background:#FFF7ED; border:1px solid #F97316; padding:8px 14px; border-radius: 20px; color:#9A3412;">
                    Bộ giải mã chi tiết và chú ý
                </span>
                <span style="color:#64748B;">&rarr;</span>
                <span style="background:#0F2942; padding:8px 14px; border-radius: 20px; color:#FFFFFF;">
                    Mặt nạ kích thước gốc (H×W)
                </span>
            </div>
        </div>
        """)

    st.caption("Thông số kiến trúc mô tả cấu hình dự án. Trạng thái mô hình hiển thị ở đầu trang; trang này không thực hiện đánh giá hiệu năng.")
    render_medical_disclaimer()
