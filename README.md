# ImgAE-Dx: Unsupervised Anomaly Detection in Medical X-ray Images

![Project Status](https://img.shields.io/badge/status-MVP%20Foundation%20Complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Tổng quan Dự án

**ImgAE-Dx** là một dự án nghiên cứu tiên phong tập trung vào **phát hiện bất thường không giám sát** trong hình ảnh X-quang y tế.

Mục tiêu cốt lõi là thực hiện một nghiên cứu so sánh chuyên sâu giữa kiến trúc **U-Net** tiêu chuẩn và một kiến trúc **Reversed Autoencoder (RA)** mới lạ.

Dự án nhằm đánh giá hiệu quả của chúng trong việc xác định các bất thường trên ảnh X-quang ngực, sử dụng bộ dữ liệu NIH Chest X-ray.

## ✨ Tính năng & Mục tiêu chính

* **Phát hiện bất thường không giám sát:** Huấn luyện mô hình chỉ trên dữ liệu "bình thường" để xác định các sai lệch.
* **Nghiên cứu so sánh:** Đánh giá hiệu suất của U-Net và Reversed Autoencoder.
* **Pipeline dữ liệu streaming:** Tích hợp trực tiếp với Kaggle API để xử lý dữ liệu hiệu quả.
* **Codebase module hóa:** Cấu trúc dự án rõ ràng, dễ bảo trì và mở rộng.
* **Khả năng tái tạo:** Tạo ra notebook Colab tự chứa từ mã nguồn sản phẩm để đảm bảo kết quả nghiên cứu có thể tái tạo.
* **Phân tích định lượng:** Sử dụng lỗi tái tạo và các chỉ số như AUC-ROC để đánh giá hiệu suất.

## 💡 Nguyên lý hoạt động

Dự án dựa trên nguyên lý của Autoencoder trong phát hiện bất thường:

1. Mô hình được huấn luyện chỉ trên hình ảnh X-quang ngực "bình thường".
2. Nó học cách tái tạo lại các hình ảnh bình thường một cách chính xác.
3. Khi một hình ảnh bất thường được đưa vào, mô hình sẽ gặp khó khăn trong việc tái tạo chính xác phần bất thường đó, dẫn đến lỗi tái tạo cao tại vị trí bất thường.
4. Bản đồ lỗi tái tạo được sử dụng để định vị và đánh giá mức độ bất thường.
