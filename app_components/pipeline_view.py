"""Visual Flowchart Component for QUY TRÌNH PHÂN TÍCH ẢNH."""

import streamlit as st


def render_pipeline_diagram():
    """Render a clean, hospital-grade flowchart of the 2-stage CADx pipeline."""
    st.html("""
        <div class="med-card" style="margin-top: 10px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0F2942; margin-bottom: 6px;">
                QUY TRÌNH PHÂN TÍCH ẢNH
            </div>
            <p style="font-size: 1rem; color: #64748B; margin-bottom: 24px;">
                Quy trình hai giai đoạn: sàng lọc ảnh MRI, sau đó phân vùng đối với kết quả dương tính.
            </p>

            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; max-width: 900px; margin: 0 auto;">
                
                <!-- Step 1: MRI Input -->
                <div style="width: 100%; background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 20px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Bước 01</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0F2942;">Ảnh MRI đầu vào</div>
                    </div>
                    <span style="font-family:var(--app-font); font-size: 1rem; color:#475569; background:#E2E8F0; padding:3px 8px; border-radius: 20px;">
                        Lát cắt 2D
                    </span>
                </div>

                <!-- Down Arrow -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 2: Preprocessing -->
                <div style="width: 100%; background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 20px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Bước 02</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0F2942;">Tiền xử lý theo mô hình</div>
                    </div>
                    <span style="font-size: 1rem; color:#475569;">
                        Chuẩn hóa Z-score • Đệm giữ tỉ lệ
                    </span>
                </div>

                <!-- Down Arrow -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 3: Swin V2-Tiny Screening -->
                <div style="width: 100%; background: #EFF6FF; border: 1px solid #93C5FD; border-left: 4px solid #0284C7; border-radius: 20px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 700; color: #0284C7; text-transform: uppercase;">Giai đoạn 1 — Sàng lọc</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0F2942;">Swin Transformer V2-Tiny</div>
                    </div>
                    <span style="font-family:var(--app-font); font-size: 1rem; color:#1E40AF; background:#DBEAFE; padding:4px 10px; border-radius: 20px;">
                        Ngưỡng xác suất = 0,40
                    </span>
                </div>

                <!-- Branching Down Arrow -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 4: Decision Split -->
                <div style="width: 100%; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <!-- Âm tính Branch -->
                    <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 20px; padding: 12px; text-align: center;">
                        <span style="font-size: 1rem; font-weight: 700; color: #059669; text-transform: uppercase;">Âm tính (P &lt; 0.40)</span>
                        <div style="font-size: 1rem; font-weight: 600; color: #065F46; margin-top: 4px;">Điểm dưới ngưỡng sàng lọc</div>
                        <div style="font-size: 1rem; color: #047857; margin-top: 2px;">Không phân vùng</div>
                    </div>

                    <!-- Dương tính Branch -->
                    <div style="background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 20px; padding: 12px; text-align: center;">
                        <span style="font-size: 1rem; font-weight: 700; color: #BE123C; text-transform: uppercase;">Dương tính (P &ge; 0.40)</span>
                        <div style="font-size: 1rem; font-weight: 600; color: #9F1239; margin-top: 4px;">Ghi nhận dấu hiệu nghi ngờ</div>
                        <div style="font-size: 1rem; color: #BE123C; margin-top: 2px;">Tiếp tục phân vùng</div>
                    </div>
                </div>

                <!-- Down Arrow into Giai đoạn 2 -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 5: Swin Hybrid Segmentation -->
                <div style="width: 100%; background: #EDF4FA; border: 1px solid #B7CADB; border-left: 4px solid #07559B; border-radius: 20px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 700; color: #07559B; text-transform: uppercase;">Giai đoạn 2 — Phân vùng</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0F2942;">Swin Hybrid (nnU-Net CNN + 2D Swin)</div>
                    </div>
                    <span style="font-family:var(--app-font); font-size: 1rem; color:#064B87; background:#E7F1FB; padding:4px 10px; border-radius: 20px;">
                        Khôi phục kích thước gốc
                    </span>
                </div>

                <!-- Down Arrow -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 6: Tumor Mask & Overlay -->
                <div style="width: 100%; background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 20px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 600; color: #64748B; text-transform: uppercase;">Kết quả đầu ra</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #0F2942;">Mặt nạ nhị phân và ảnh chồng lớp</div>
                    </div>
                    <span style="font-size: 1rem; color:#475569;">
                        Diện tích • Khung giới hạn • Tỷ lệ
                    </span>
                </div>

                <!-- Down Arrow -->
                <div style="color: #94A3B8; font-size: 16px; line-height: 1;">↓</div>

                <!-- Step 7: AI Analysis Result -->
                <div style="width: 100%; background: #0F2942; border-radius: 20px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; color: #FFFFFF;">
                    <div>
                        <span style="font-size: 1rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Hoàn thành</span>
                        <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF;">Kết quả và tóm tắt phân tích AI</div>
                    </div>
                    <span style="font-size: 1rem; color:#38BDF8; font-weight:600;">
                        Báo cáo phân tích
                    </span>
                </div>

            </div>
        </div>
        """)
