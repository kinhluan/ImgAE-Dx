# QEC

## 1. Question

Làm thế nào để xây dựng một Notebook trình bày toàn bộ quá trình phát triển mô hình Reversed Autoencoder (RA) cho bài toán phát hiện bất thường không giám sát trong ảnh y khoa?

## 2. Evidence

Bộ dữ liệu thử nghiệm :

1. MRI não (u não, đột quỵ)
2. X-quang cổ tay trẻ em (gãy xương)
3. X-quang ngực (bệnh phổi như lao)

Chỉ số đánh giá :

1. AUC (Area Under Curve)
2. Độ chính xác định vị (pixel-level localization)
3. Độ nhạy (sensitivity)

## 3. Conclusion

Phát triển source code -> Notebook Python hoàn chỉnh với các phần sau

1. Giới thiệu
    * Mô tả bài toán phát hiện bất thường không giám sát trong ảnh y khoa
    * Trình bày mục tiêu: xây dựng một mô hình tổng quát cho nhiều dạng ảnh y khoa
    * Giới thiệu dataset: liệt kê các loại ảnh sử dụng (MRI, X-ray, v.v.), nguồn gốc và đặc điểm chính

2. Tiền xử lý dữ liệu
    * Làm sạch dữ liệu
    * Chuẩn hóa
    * Tạo tập train/test :
    * Chỉ dùng ảnh “bình thường” cho tập train (unsupervised).
    * Tập test gồm cả ảnh bình thường và bất thường để đánh giá

3. Xây dựng mô hình
4. Training & Evaluation
5. Analysis
6. Conclusion
