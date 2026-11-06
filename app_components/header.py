"""Vietnamese research application header."""
from html import escape
import streamlit as st
from app_services.inference_service import engine


def render_header(title="HỆ THỐNG PHÂN TÍCH MRI NÃO", subtitle="Hỗ trợ sàng lọc và phân vùng bằng trí tuệ nhân tạo"):
    status = "Mô hình sẵn sàng" if engine.is_real_loaded else "Lỗi tải mô hình" if engine.mode == "real" else "Chế độ mô phỏng"
    st.html(f"""<div class="hospital-masthead">ỨNG DỤNG NGHIÊN CỨU HÌNH ẢNH Y HỌC <span>SWIN V2 · SWIN HYBRID</span></div>
    <div class="clinical-header"><div><h1 class="clinical-title">{escape(title)}</h1>
    <p class="clinical-subtitle">{escape(subtitle)}</p></div>
    <div class="system-status"><div>TRẠNG THÁI HỆ THỐNG</div><strong>{status}</strong></div></div>""")
    if not engine.is_real_loaded:
        if engine.mode == "real":
            st.warning("Chưa tải được mô hình thật. Chức năng suy luận đang tạm ngừng.")
        else:
            st.info("Đang dùng chế độ mô phỏng để thử chức năng. Kết quả không có ý nghĩa chẩn đoán.")
        if engine.error:
            with st.expander("Thông tin tải mô hình"):
                if "No module named" in engine.error:
                    st.write("Môi trường Python đang thiếu thư viện. Cài requirements.txt bằng đúng Python dùng để chạy ứng dụng.")
                elif "torchvision" in engine.error:
                    st.write("Thư viện PyTorch và torchvision không tương thích hoặc cài đặt chưa hoàn chỉnh. Hãy chạy ứng dụng bằng môi trường .venv của dự án.")
                elif "Git LFS" in engine.error:
                    st.write("Cần tải tệp trọng số đầy đủ; tệp con trỏ Git LFS không chứa trọng số.")
                else:
                    st.write("Không khởi tạo được mô hình. Xem chi tiết lỗi bên dưới để kiểm tra thư viện và cấu hình checkpoint.")
                st.code(engine.error, language=None)
