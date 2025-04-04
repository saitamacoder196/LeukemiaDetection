# Phát hiện bệnh qua ảnh tế bào

Dự án luận văn thạc sĩ về phát hiện bệnh ung thư máu qua ảnh tế bào sử dụng các kỹ thuật Deep Learning.

## Giới thiệu

Dự án này tập trung vào việc huấn luyện các mô hình học sâu để phát hiện bệnh bạch cầu (Leukemia) thông qua phân tích ảnh tế bào máu. Sử dụng các kiến trúc CNN hiện đại như ResNetRS50, EfficientNet, Inception_V3, Xception và các mô hình khác để phân loại tế bào máu thành 4 loại: Benign (tế bào lành tính), Early Pre B, Pre B, và Pro B (các giai đoạn khác nhau của bệnh).

## Cấu trúc thư mục

- `data/`: Chứa dữ liệu
  - `raw/`: Dữ liệu gốc
    - `benign/`: Tế bào lành tính
    - `early_pre_b/`: Giai đoạn Early Pre B
    - `pre_b/`: Giai đoạn Pre B
    - `pro_b/`: Giai đoạn Pro B
  - `processed/`: Dữ liệu đã được tiền xử lý
  - `augmented/`: Dữ liệu đã được tăng cường
- `models/`: Chứa các mô hình học sâu
- `utils/`: Tiện ích hỗ trợ
- `notebooks/`: Jupyter notebooks
- `scripts/`: Scripts thực thi
- `results/`: Kết quả thí nghiệm
- `docs/`: Tài liệu

## Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/LeukemiaDetection.git
cd LeukemiaDetection

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

## Sử dụng

### Huấn luyện mô hình

```bash
python scripts/train.py --model resnet --epochs 100 --batch_size 32
```

### Đánh giá mô hình

```bash
python scripts/evaluate.py --model resnet --weights path/to/weights
```

### Dự đoán

```bash
python scripts/predict.py --model resnet --weights path/to/weights --image path/to/image
```

## Tác giả

Nguyễn Hồng Phát - MSHV: M5123006

## Giảng viên hướng dẫn

- PGS.TS. Nguyễn Thanh Hải
- TS. Trần Thanh Điện
