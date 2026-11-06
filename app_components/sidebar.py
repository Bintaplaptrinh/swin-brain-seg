"""Sidebar Navigation Component for Brain Tumor CADx."""

import streamlit as st


def render_sidebar() -> str:
    """Render clinical sidebar with branding, navigation links, and system disclaimer.

    Returns:
        The selected page name string.
    """
    with st.sidebar:
        # Branding
        st.html("""
            <div style="padding: 12px 0 20px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 46px; min-width: 46px; height: 46px; background: #0F2942; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 800; font-size: 18px; letter-spacing: -0.05em;">
                        MRI
                    </div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0F2942; line-height: 1.15; letter-spacing: -0.02em;">
                            PHÂN TÍCH MRI NÃO<br><span style="color: #0284C7;">CADx</span>
                        </div>
                    </div>
                </div>
            </div>
            """)

        if "active_page" not in st.session_state:
            st.session_state.active_page = "Tổng quan"

        def navigate(page):
            st.session_state.active_page = page

        st.html('<p class="nav-group-title">KHÔNG GIAN PHÂN TÍCH</p>')
        with st.container(key="primary_navigation"):
            st.button(
                "Phân tích MRI", key="nav_analysis", type="primary",
                use_container_width=True, on_click=navigate, args=("Phân tích MRI",),
            )
            st.caption("Tải ảnh · Sàng lọc · Phân vùng")
            if st.session_state.active_page == "Phân tích MRI":
                st.html('<span class="nav-active-note">Đang mở không gian phân tích</span>')

        st.html('<p class="nav-group-title nav-information">THÔNG TIN THAM KHẢO</p>')
        with st.container(key="information_navigation"):
            for page, key in [("Tổng quan", "overview"), ("Thông tin mô hình", "models"), ("Giới thiệu", "about")]:
                st.button(
                    page, key="nav_" + key, use_container_width=True,
                    type="primary" if st.session_state.active_page == page else "secondary",
                    on_click=navigate, args=(page,),
                )
        selected_page = st.session_state.active_page

        st.markdown("---")

        # System Specs summary in sidebar
        st.html("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 20px; padding: 12px 14px; margin-bottom: 20px;">
                <div style="font-size: 1rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Trạng thái hệ thống</div>
                <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981;"></span>
                    <span style="font-size: 1rem; font-weight: 600; color: #0F172A;">Ứng dụng nghiên cứu</span>
                </div>
                <div style="font-size: 1rem; color: #64748B; margin-top: 4px;">Chế độ hoạt động hiển thị tại đầu trang</div>
            </div>
            """)

        # Mandatory Footer Disclaimer
        st.html("""
            <div style="padding-top: 10px; font-size: 1rem; color: #94A3B8; line-height: 1.45; text-align: left;">
                <strong style="color: #64748B;">HỆ THỐNG AI NGHIÊN CỨU</strong><br>
                Phục vụ nghiên cứu và đào tạo.<br>
                <em>Không thay thế chẩn đoán của bác sĩ.</em>
            </div>
            """)

    return selected_page
