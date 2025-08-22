# ImgAE-Dx: Unsupervised Anomaly Detection in Medical X-ray Images

![Project Status](https://img.shields.io/badge/status-MVP%20Foundation%20Complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Tổng quan

**ImgAE-Dx - Image Autoencoder Diagnosis Anomaly Research** là một dự án tập trung vào **phát hiện bất thường không giám sát** trong hình ảnh X-quang y tế. Mục tiêu là thực hiện một nghiên cứu so sánh chuyên sâu giữa kiến trúc **U-Net** tiêu chuẩn và một kiến trúc **Reversed Autoencoder (RA)** mới. Dự án nhằm đánh giá hiệu quả trong việc xác định các bất thường trên ảnh X-quang, sử dụng bộ dữ liệu NIH Chest X-ray (hoặc các datasets khác).

## Mục tiêu chính

* **Phát hiện bất thường không giám sát:** Huấn luyện mô hình chỉ trên dữ liệu "bình thường" để xác định các sai lệch.
* **Nghiên cứu so sánh:** Đánh giá hiệu suất của U-Net và Reversed Autoencoder.
* **Phân tích định lượng:** Sử dụng lỗi tái tạo và các chỉ số như AUC-ROC để đánh giá hiệu suất.

## Nguyên lý hoạt động

Dự án dựa trên nguyên lý của Autoencoder trong phát hiện bất thường:

1. Mô hình được huấn luyện chỉ trên hình ảnh X-quang ngực "bình thường".
2. Học cách tái tạo lại các hình ảnh bình thường một cách chính xác.
3. Khi một hình ảnh bất thường được đưa vào, mô hình sẽ gặp khó khăn trong việc tái tạo chính xác phần bất thường đó, dẫn đến lỗi tái tạo cao tại vị trí bất thường.
4. Bản đồ lỗi tái tạo được sử dụng để định vị và đánh giá mức độ bất thường.

## 🚀 Chạy nhanh trên Google Colab

**Muốn chạy thử ngay? Chỉ cần 3 bước:**

1. **[📖 Đọc hướng dẫn chi tiết](docs/guides/GOOGLE_COLAB_VI.md)** - Hướng dẫn từng bước bằng tiếng Việt
2. **[📓 Mở notebook](https://colab.research.google.com/github/kinhluan/ImgAE-Dx/blob/main/notebooks/T4_GPU_Training_Colab.ipynb)** - Click để mở trực tiếp trên Colab  
3. **▶️ Chạy tất cả cell** - Ngồi chờ kết quả (45-90 phút)

🎯 **Kết quả:** Hai mô hình AI phát hiện bất thường + phân tích so sánh chi tiết

## 📚 Documentation

Tài liệu dự án được tổ chức trong thư mục [`docs/`](docs/):

### 🚀 Getting Started

- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Hướng dẫn nhanh để bắt đầu
- **[🇻🇳 Hướng dẫn Google Colab](docs/guides/GOOGLE_COLAB_VI.md)** - Cách chạy trên Google Colab (Tiếng Việt)
- **[Training Guide](docs/guides/TRAINING_GUIDE.md)** - Chi tiết cách train U-Net và Reversed AE

### 🏗️ Technical Documentation  

- **[Architecture Overview](docs/architecture/ARCHITECTURE.md)** - Kiến trúc tổng thể
- **[Model Architecture](docs/architecture/IMG_AE_DX_ARCHITECTURE.md)** - Chi tiết models

### 🔬 Research & Development

- **[Research Journey](docs/research/PROJECT_JOURNEY.md)** - Quá trình nghiên cứu
- **[Ideas & Concepts](docs/research/IDEA.md)** - Background và lý thuyết
- **[Development Progress](docs/development/PROJECT_COMPLETION.md)** - Tiến độ phát triển

📖 **[Xem tất cả documentation](docs/README.md)**
