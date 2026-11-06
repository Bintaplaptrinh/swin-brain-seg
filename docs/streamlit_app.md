# Streamlit research dashboard

## Chạy ứng dụng

Tại thư mục gốc, với Python 3.10 trở lên:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Hoặc chạy `run_app.bat`. Theme sáng được cấu hình trong `.streamlit/config.toml`.

## Chế độ thực thi

```powershell
$env:MODEL_MODE = "mock"  # auto (mặc định), mock, hoặc real
python -m streamlit run app.py
```

- `mock`: thuật toán cường độ ảnh để thử giao diện; không chạy Swin và không có ý nghĩa y khoa.
- `auto`: tải pipeline PyTorch nếu có đủ checkpoint. Nếu thiếu hoặc tải lỗi, header hiển thị lý do và thông báo mô phỏng.
- `real`: nếu không tải được model, vô hiệu hóa inference và hiển thị lỗi; không thay bằng kết quả giả.
- Lỗi khi model thật đang suy luận luôn được báo; không tự chuyển sang mock.

Khởi động lại Streamlit sau khi thay checkpoint hoặc biến môi trường.

## Workflow

1. MRI Analysis → upload PNG/JPG/JPEG hoặc chọn Synthetic examples → Load example.
2. Run screening; ngưỡng mặc định 0.40 từ cấu hình dự án.
3. Dương tính → Run Tumor Segmentation. Âm tính → bỏ qua segmentation.
4. Xem ảnh gốc, mask, overlay; đổi opacity và bounding box.
5. Tải mask PNG, overlay PNG, báo cáo JSON ghi nguồn ảnh và chế độ thực thi.

Đổi ảnh xóa kết quả cũ; đổi opacity hoặc chuyển trang giữ kết quả. Kết quả phân vùng được xem ngay trong trang Phân tích MRI; không còn mục Phân vùng riêng. Các ví dụ là hình tổng hợp sinh xác định, không phải dữ liệu BRISC hay ảnh bệnh nhân; mask tham chiếu không được dùng làm dự đoán. Contrast enhancement chỉ đổi ảnh hiển thị, không phải CLAHE và không tác động inference.

Diện tích tính theo pixel ở kích thước ảnh gốc; tỷ lệ dùng toàn ảnh làm mẫu số. Bounding box là `(x1,y1,x2,y2)` với hai đầu bao hàm. Số vùng dùng thành phần liên thông 4 hướng, không phải số khối u hay thể tích.

## Tích hợp checkpoint thật

Adapter `app_services/inference_service.py` sử dụng `src.pipeline.CADxPipeline` và model hiện có, không xây kiến trúc thay thế. Cần binary weights thực, không phải file con trỏ Git LFS:

- `outputs/checkpoints/classifer_swinv2_tiny/classifer_swinv2_tiny_binary.pt`
- `outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/last.pt`
- `outputs/checkpoints/pretrained/Brain_Swin_UNETR.pth` (tokenizer được checkpoint segmentation tham chiếu)

Tải các file Git LFS của repository hoặc đặt checkpoint tương thích tại các đường dẫn này. Với checkpoint khác, cập nhật hằng đường dẫn trong adapter. Pipeline kiểm tra state dict và tokenizer hash nếu checkpoint có lưu hash. Chế độ ứng dụng hiện sử dụng PyTorch; các bản ONNX dành cho luồng inference khác trong repository.

Screening gọi `classifier.predict_slice` riêng, không chạy segmentation sớm. Segmenter sử dụng tiền xử lý letterbox, chuẩn hóa nonzero và khôi phục xác suất về kích thước gốc theo pipeline hiện có. Mask nhị phân mặc định dùng ngưỡng 0.50. Không coi thống kê thí nghiệm cũ là hiệu năng đã xác minh của ứng dụng đang chạy.

## Tổ chức mã

- `app.py`: entry point và điều hướng.
- `app_components/`: sidebar, header, cards, sơ đồ pipeline.
- `app_views/`: overview, analysis, segmentation, model information, about.
- `app_services/inference_service.py`: adapter model thật và mock.
- `app_utils/`: xử lý ảnh, đo mask, overlay, CSS.
- `src/`: model và pipeline nghiên cứu gốc.
- `tests/test_app_workflow.py`: kiểm tra gating, trạng thái, mock và các trang bằng Streamlit AppTest.

```powershell
python -m pytest tests/test_app_workflow.py -q
```

Research Use Only — ứng dụng không thay thế chẩn đoán chuyên môn.


## Môi trường Windows riêng cho ứng dụng

Khi Python toàn máy có xung đột torch/torchvision, dùng môi trường dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MODEL_MODE = "real"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`run_app.bat` ưu tiên Python trong `.venv` và mặc định chế độ `real` nếu chưa đặt `MODEL_MODE`. Dừng tiến trình Streamlit cũ trước khi mở lại để nạp đúng thư viện và model.
