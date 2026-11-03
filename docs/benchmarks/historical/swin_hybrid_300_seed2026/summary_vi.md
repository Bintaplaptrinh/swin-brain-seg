**Kết quả thí nghiệm Brain Swin + nnU-Net kernel depthwise 9×9**

Đã hoàn tất lúc 05:05:10 ngày 02/11/2026 (UTC+7), sau 3 giờ 35 phút 21 giây tính từ lúc khởi chạy supervisor. Train dừng đúng early stopping ở epoch 103/300: 30 epoch liên tiếp không cải thiện validation Glioma ít nhất 0,0001 kể từ mốc tốt nhất. Checkpoint được chọn là **student epoch 73**, dựa hoàn toàn vào validation Glioma mIoU. Test chỉ được đánh giá sau khi chọn checkpoint.

**Glioma test mIoU = 0.673706, chưa đạt mục tiêu ≥ 0,8**; còn thiếu 0.126294 điểm IoU. Weighted mIoU 0.795115 là kết quả gộp các lớp, không phải Glioma mIoU.

| Nhóm | Số ảnh test | Validation mIoU tại best | Test mIoU | Test Dice |
|---|---:|---:|---:|---:|
| Glioma | 254 | 0.733687 | 0.673706 | 0.764650 |
| Meningioma | 306 | 0.886693 | 0.897782 | 0.942065 |
| Pituitary | 300 | 0.808719 | 0.793186 | 0.872944 |
| Toàn bộ, trọng số theo số ảnh | 860 | 0.813146 | 0.795115 | 0.865554 |

Weight EMA dự kiến bắt đầu ở epoch 150 nên **chưa từng được kích hoạt** trong lượt này. Efficient Multi-Scale Attention tại skip connections vẫn hoạt động từ đầu; đó là cơ chế khác với weight EMA. Lượt này chưa đánh giá được lợi ích của weight EMA từ 150, giai đoạn LR cuối của lịch 300 epoch, hoặc mức phạt biên tối đa tại epoch 150. Không tự kéo dài train sau khi đã dừng đúng lịch.

| Thí nghiệm đã lưu | Test Glioma | Test Meningioma | Test Pituitary | Weighted mIoU |
|---|---:|---:|---:|---:|
| biref_seed2026 | 0.682290 | 0.900794 | 0.805507 | 0.803019 |
| biref_ema_rl_seed2026 | 0.666913 | 0.891664 | 0.775279 | 0.784685 |
| biref_RL_seed2026 | 0.679133 | 0.900543 | 0.799613 | 0.799942 |
| Swin hybrid hiện tại | 0.673706 | 0.897782 | 0.793186 | 0.795115 |

So với biref_ema_rl, Glioma tăng 0.679 điểm phần trăm. So với BiRef ban đầu, Glioma giảm 0.858 điểm phần trăm và weighted mIoU giảm 0.790 điểm phần trăm. Tăng dung lượng lên 38,87M chưa đem lại cải thiện Glioma so với BiRef 8,86M trong lượt này.

Đã xác minh các báo cáo trên dùng cùng danh sách validation và 860 ảnh test. Tuy nhiên đây không phải ablation hoàn toàn tương đương: lượt mới đổi dung lượng, pretrained transfer, kernel, loss, augmentation và lịch train; đồng thời tắt RL và trả policy holdout về tập fit. Vì vậy không thể quy nguyên nhân kết quả cho riêng Swin hoặc riêng boundary loss.

Test toàn bộ: Dice 0.865554, recall 0.870939, precision 0.903133, HD95 17.4134 px, ASSD 5.2052 px, NSD 0.722649 với tolerance 2 px. Không có dự đoán rỗng hay khoảng cách vô hạn ở checkpoint được chọn trên test. So với BiRef, NSD nhỉnh hơn nhưng HD95/ASSD kém hơn; chưa thể nói toàn bộ chất lượng biên đã cải thiện.

Phân tích riêng Glioma cho thấy các mặt cắt toàn bộ có thể khá đều nhưng Glioma vẫn chênh đáng kể:

| Glioma trên test | Số ảnh | mIoU | Dice | Số ảnh IoU = 0 | Số ảnh IoU < 0,5 |
|---|---:|---:|---:|---:|---:|
| Axial | 85 | 0.603090 | 0.689932 | 9 | 29 |
| Coronal | 81 | 0.696994 | 0.789371 | 3 | 17 |
| Sagittal | 88 | 0.720478 | 0.814067 | 1 | 15 |

Tổng cộng 61/254 ảnh Glioma có IoU < 0,5; 13 ảnh có IoU = 0 dù prediction không rỗng. Axial là nhóm yếu nhất. Micro recall Glioma 0.760609 thấp hơn micro precision 0.840955, cho thấy bỏ sót vùng u vẫn đáng kể. HD95 riêng Glioma 32.3915 px, ASSD 11.5601 px, NSD 0.532934. Đây là các nhóm lỗi cần kiểm tra trực quan, chưa phải bằng chứng rằng nguyên nhân là thiếu số mẫu hay nhãn sai. Các phân tích test này chỉ dùng báo cáo, không dùng điều chỉnh lượt train đã kết thúc.

Mô hình có **38.874.750** tham số: **36.885.900 trainable**, **1.988.850 frozen**. nnU-Net ResEnc dùng các khối residual 3×3 và context depthwise 9×9 dilation 1 tại E3–E6. Checkpoint Brain_Swin_UNETR.pth nguồn là 3D, ba kênh; Swin được chuyển 3D→2D grayscale và đóng băng, rồi ghép vào encoder nnU-Net qua adapters trainable. 126 tensor Swin được tải/biến đổi có kiểm tra; 109 giữ nguyên. Không sử dụng convolution encoder/decoder của checkpoint nguồn. Không xác minh được dữ liệu/modalities đã dùng pretrain chỉ từ state dictionary.

Tập fit 3.146 ảnh (917 Glioma), validation 787 ảnh, test 860 ảnh; ảnh vào 256×256. Robust augmentation cho tất cả nhóm, sampling weight Glioma cố định 1,5, RL tắt. AdamW LR 0,0003, weight decay 0,0001, FP16 AMP, CosineAnnealingLR. Loss động alpha*Dice + beta*Focal + gamma*Boundary, với alpha=1; beta tăng 0,5→1 và gamma tăng 0,5→2 tới epoch 150, sau đó giữ nguyên. Padding không góp vào loss. Tại best epoch 73: beta=0,741611, gamma=1,224832; tại epoch 103: beta=0,842282, gamma=1,526846.

Batch vật lý/effective **16/16**, **4 workers** trong toàn bộ 103 epoch. VRAM PyTorch peak allocated **4.467 GB**, reserved **4.830 GB**, dưới ngân sách 9 GB; số này không bao gồm CUDA context và ứng dụng khác. Không có OOM, lỗi worker hoặc lần tự phục hồi tài nguyên; 20,288 optimizer updates thành công và 3 updates được AMP bỏ qua do overflow. Checkpoint mỗi epoch, lịch scheduler, số epoch và early stop đã đối chiếu đúng history. Supervisor và trainer đã thoát exit code 0; GPU đã được giải phóng.

Suy luận forward-only trên RTX 3060 ở batch 1, FP16, 256×256: median 27.30 ms/ảnh. Không bao gồm đọc ảnh, tiền xử lý, trả về kích thước gốc hoặc tính metrics.

Báo cáo nguồn: `test_metrics.json`, `best_validation.json`, `test_per_image.csv`, `history.jsonl`, `tokenizer_transfer.json`, `completion_audit.json`. Mô hình chọn cuối: `best.pt`; trạng thái cuối để khôi phục/kiểm tra: `last.pt`. Lịch giám sát được xóa sau khi báo hoàn tất để tránh thông báo lặp.

![Validation và so sánh test](D:/Project/Brain/runs/swin_hybrid_300_seed2026/training_curves.png)
