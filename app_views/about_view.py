"""Project information in Vietnamese."""
import streamlit as st
from app_components.header import render_header
from app_components.cards import render_medical_disclaimer


def render_about_page():
    render_header("Giới thiệu hệ thống", "Nghiên cứu ứng dụng trí tuệ nhân tạo trong phân tích ảnh MRI não.")
    with st.container(border=True):
        st.subheader("Thông tin đề tài")
        st.write("Hệ thống CADx sử dụng Swin V2-Tiny để sàng lọc từng lát cắt MRI não và Swin Hybrid để dự đoán vùng nghi ngờ u. Dự án nghiên cứu trên bộ dữ liệu BRISC 2025.")
        st.write("Swin Hybrid kết hợp bộ mã hóa CNN phần dư nnU-Net với Swin Transformer 2D, hợp nhất đặc trưng đa tỉ lệ và bộ giải mã chú ý.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Nhóm thực hiện")
        st.markdown("Trần Mạnh Hùng  \nNguyễn Phước Toàn  \nNguyễn Hữu Tuấn Phát")
        st.markdown("**Giảng viên hướng dẫn:** Hoàng Thị Kiều Anh")
    with right, st.container(border=True):
        st.subheader("Dữ liệu và tài liệu tham khảo")
        st.markdown("[Bộ dữ liệu BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)")
        st.write("Swin Transformer (Liu và cộng sự, 2021); nnU-Net (Isensee và cộng sự, 2021); thư viện MONAI.")
    render_medical_disclaimer()
