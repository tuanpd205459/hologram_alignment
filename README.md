# Hologram Alignment

Dự án này cung cấp các công cụ xử lý và căn chỉnh ảnh hologram off-axis. Thuật toán hỗ trợ lọc phổ tự động để lấy phổ bậc +1 và tìm ra độ lệch phase lý tưởng (sub-pixel shift) để bù pha giữa hai ảnh.

## Cấu trúc thư mục
- `src/`: Chứa mã nguồn cốt lõi (`core.py`).
- `tests/`: Chứa các kịch bản kiểm thử thuật toán bằng dữ liệu giả lập (`test_alignment.py`).
- `main.py`: Điểm đầu vào của chương trình, ví dụ cách chạy cho một file ảnh thực tế.

## Cài đặt
Yêu cầu Python >= 3.8. Bạn có thể cài đặt thư viện qua lệnh:
```bash
pip install -r requirements.txt
```

## Sử dụng
Sử dụng gói `src.core` để lấy các hàm như `align_two_holograms` hoặc `extract_plus_one_order`. Bạn có thể tham khảo tệp `main.py` để xem ví dụ.
