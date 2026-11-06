"""Explicit, persistent two-stage research workflow."""
import hashlib
import io
import streamlit as st
from PIL import Image, UnidentifiedImageError
from app_components.header import render_header
from app_components.cards import render_medical_disclaimer
from app_services.inference_service import engine
from app_utils.imaging import BENCHMARK_PRESETS, generate_benchmark_slice, load_mri_image
from app_views.segmentation_view import render_results


def set_input(image, identity, metadata):
    if st.session_state.get("input_identity") != identity:
        st.session_state.update(current_image=image, input_identity=identity,
                                image_meta=metadata, screening_result=None, segmentation_result=None)


def render_analysis_page():
    render_header("Phân tích MRI", "Tải ảnh MRI lát cắt 2D, thực hiện sàng lọc và phân vùng khi có kết quả dương tính.")
    source = st.radio("Nguồn ảnh", ["Tải ảnh MRI", "Ảnh mô phỏng"], horizontal=True)
    if source == "Tải ảnh MRI":
        uploaded = st.file_uploader("Chọn ảnh MRI (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            data = uploaded.getvalue()
            try:
                with Image.open(io.BytesIO(data)) as raw:
                    if raw.width * raw.height > 16_000_000:
                        raise ValueError("Vui lòng sử dụng ảnh 2D không quá 16 triệu điểm ảnh.")
                    image = load_mri_image(raw)
                    metadata = {"filename": uploaded.name, "format": raw.format,
                                "dimensions": f"{image.width} × {image.height}", "source": "Ảnh tải lên"}
                set_input(image, hashlib.sha256(data).hexdigest(), metadata)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                st.error(f"Không đọc được ảnh: {exc}")
                render_medical_disclaimer()
                return
    else:
        st.info("Ảnh được tạo để thử chức năng, không phải ảnh bệnh nhân hoặc dữ liệu đánh giá BRISC.")
        choice = st.selectbox("Chọn ảnh mẫu", list(BENCHMARK_PRESETS), format_func=lambda k: BENCHMARK_PRESETS[k]["name"])
        if st.button("Nạp ảnh mẫu"):
            image, _ = generate_benchmark_slice(choice)
            set_input(image, "synthetic:" + choice, {"filename": choice + "_synthetic.png",
                      "format": "PNG", "dimensions": f"{image.width} × {image.height}", "source": "Ảnh mô phỏng"})
    image = st.session_state.get("current_image")
    if image is None:
        st.info("Vui lòng tải ảnh MRI hoặc nạp ảnh mô phỏng để bắt đầu.")
        render_medical_disclaimer()
        return
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Ảnh MRI đầu vào")
        st.image(image, use_container_width=True)
        meta = st.session_state["image_meta"]
        st.caption(f"{meta['filename']} · {meta['dimensions']} px · {meta['format']} · {meta['source']}")
        st.caption("Đã đọc ảnh thang xám. Việc chuẩn hóa và đổi kích thước được thực hiện khi chạy mô hình.")
    with right:
        st.subheader("Sàng lọc bằng AI")
        if st.button("Thực hiện sàng lọc", type="primary"):
            st.session_state.update(screening_result=None, segmentation_result=None)
            try:
                with st.spinner("Đang thực hiện sàng lọc..."):
                    st.session_state.screening_result = engine.predict_classification(image)
            except Exception as exc:
                st.error(f"Sàng lọc không thành công: {exc}")
        result = st.session_state.get("screening_result")
        if result:
            st.caption(result["execution_mode"])
            st.metric("Điểm sàng lọc" if not engine.is_real_loaded else "Xác suất có u", f"{result['probability']:.1%}")
            st.caption(f"Ngưỡng: {result['threshold']:.2f} · {result['model_name']}")
            if result["is_positive"]:
                st.warning("Sàng lọc dương tính — hệ thống ghi nhận dấu hiệu nghi ngờ.")
                if st.button("Thực hiện phân vùng"):
                    try:
                        with st.spinner("Đang thực hiện phân vùng..."):
                            st.session_state.segmentation_result = engine.predict_segmentation(image)
                    except Exception as exc:
                        st.error(f"Phân vùng không thành công: {exc}")
            else:
                st.info("Sàng lọc âm tính — điểm dưới ngưỡng. Không thực hiện phân vùng.")
            if not engine.is_real_loaded:
                st.caption("Điểm và vùng dự đoán mô phỏng chỉ dựa trên cường độ ảnh, không có ý nghĩa y khoa.")
    if st.session_state.get("segmentation_result"):
        render_results()
    render_medical_disclaimer()
