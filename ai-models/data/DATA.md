# Mô tả dữ liệu

## Nguồn
- **Tên dataset:** Mobile Price Classification
- **Tác giả:** Abhishek Sharma
- **Link:** https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
- **Giấy phép:** Unkown
- **Ngày tải:** 24/09/2026

## Nội dung file
- `dataset.zip` chứa `train.csv` (dùng để huấn luyện và đánh giá).
- Không dùng `test.csv` của Kaggle vì file này không có cột nhãn.

## Thống kê
- Số mẫu: 2000
- Số đặc trưng: 20 (14 số liên tục/rời rạc, 6 nhị phân)
- Cột mục tiêu: `price_range` (0 = Low Cost, 1 = Medium Cost, 2 = High Cost, 3 = Very High Cost)
- Phân bố lớp: <điền số mẫu mỗi lớp từ bước kiểm tra>
- Giá trị thiếu: <điền>

## Bài toán
Phân loại đa lớp (4 lớp): dự đoán khoảng giá của điện thoại từ thông số phần cứng.

## Cách giải nén để chạy
    unzip -o ai-models/data/dataset.zip -d ai-models/data