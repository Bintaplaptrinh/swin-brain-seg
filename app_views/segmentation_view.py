"""Read-only viewer for a completed, positively gated analysis."""
import io
import json
import streamlit as st
from app_components.cards import render_clinical_summary_card
from app_utils.imaging import apply_contrast_enhancement, create_clinical_overlay


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_results():
    image = st.session_state.current_image
    result = st.session_state.segmentation_result
    screening = st.session_state.screening_result
    mask = result["mask"]
    metrics = result["metrics"]
    st.subheader("Kết quả phân vùng u não")
    st.caption(result["execution_mode"])
    opacity = st.slider("Độ đậm lớp phủ", 0.0, 1.0, 0.5, 0.05)
    draw_box = st.checkbox("Hiện khung giới hạn", value=True)
    enhance = st.checkbox("Tăng tương phản khi xem", value=False)
    base = apply_contrast_enhancement(image) if enhance else image
    overlay = create_clinical_overlay(base, mask, opacity=opacity, draw_bbox=draw_box)
    for col, title, img in zip(st.columns(3), ["Ảnh MRI gốc", "Mặt nạ phân vùng", "Ảnh chồng lớp"], [image, mask, overlay]):
        with col:
            st.markdown(f"**{title}**")
            st.image(img, use_container_width=True)
    st.caption(f"Mô hình: {result['model_name']} · Khung giới hạn (x1, y1, x2, y2; gồm điểm biên): {metrics['bbox']}")
    st.caption("Tỷ lệ diện tích được tính trên toàn ảnh. Số vùng là số thành phần liên thông của mặt nạ, không phải số khối u.")
    render_clinical_summary_card(
        screening_result="Dương tính" if screening["is_positive"] else "Âm tính",
        probability=screening["probability"] * 100,
        segmentation_status="Hoàn tất" if result["tumor_detected"] else "Hoàn tất — mặt nạ rỗng",
        detected_regions=f"{metrics['region_count']} vùng",
        tumor_area=metrics["tumor_pixels"], tumor_ratio=metrics["tumor_ratio_percent"],
    )
    report = {"source": st.session_state.image_meta, "screening": screening,
              "segmentation": {k: v for k, v in result.items() if k != "mask"},
              "research_use_only": True}
    a, b, c = st.columns(3)
    a.download_button("Tải mặt nạ", png_bytes(mask), "mask.png", "image/png")
    b.download_button("Tải ảnh chồng lớp", png_bytes(overlay), "overlay.png", "image/png")
    c.download_button("Tải báo cáo", json.dumps(report, indent=2, ensure_ascii=False), "analysis.json", "application/json")

