# Paper: Towards Universal Unsupervised Anomaly Detection in Medical Imaging

## Paper

* <https://paperswithcode.com/paper/towards-universal-unsupervised-anomaly>
* <https://arxiv.org/abs/2401.10637v1>

## Source code

* <https://github.com/ci-ber/RA>

## Vấn đề

Các phương pháp phát hiện bất thường không giám sát hiện tại trong hình ảnh y khoa thường chỉ hiệu quả với một số loại bệnh lý cụ thể (ví dụ: tổn thương trong MRI não) và gặp khó khăn khi xử lý các bất thường đa dạng hoặc trên các phương thức hình ảnh khác nhau (như X-quang, CT).

## Mục tiêu

Đề xuất một phương pháp tổng quát, có thể phát hiện nhiều loại bệnh lý trên nhiều phương thức hình ảnh y khoa (MRI não, X-quang cổ tay trẻ em, X-quang ngực) mà không cần dữ liệu huấn luyện có nhãn (unsupervised).

## Dữ liệu huấn luyện

Mô hình Reversed Autoencoder (RA) được huấn luyện chỉ trên dữ liệu hình ảnh bình thường (healthy images) để học cách tái tạo các hình ảnh không có bất thường. Trong giai đoạn kiểm tra, mô hình so sánh hình ảnh gốc (có thể chứa bất thường) với hình ảnh tái tạo giả lành tính (pseudo-healthy) để phát hiện bất thường

## Xây dựng mô hình

RA, một biến thể của Autoencoder được thiết kế đặc biệt để tái tạo hình ảnh giả lành tính (pseudo-healthy reconstructions)

## Ứng dụng

1. Mô hình RA được sử dụng để phát hiện bất thường trong hình ảnh y khoa, bao gồm MRI não (u não, đột quỵ), X-quang cổ tay trẻ em (gãy xương), và X-quang ngực (bệnh lý phổi như lao)
2. Phương pháp phát hiện bất thường dựa trên lỗi tái tạo: So sánh hình ảnh gốc với hình ảnh tái tạo giả lành tính để xác định các vùng khác biệt (bất thường)

## Review

1. AUC (Area Under the Curve)
a. Đo hiệu suất phân loại bất thường so với bình thường.
2. Độ chính xác định vị bất thường (pixel-level localization)
a. Đánh giá khả năng xác định chính xác vị trí bất thường trên ảnh
3. Độ nhạy (sensitivity)
a. Đo tỷ lệ phát hiện đúng các bất thường
