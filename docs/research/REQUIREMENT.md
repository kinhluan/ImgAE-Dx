# Ứng dụng kiến trúc Autoencoder

## Nội dung

* Nhóm bắt buộc phải sử dụng một kiến trúc Autoencoder hoặc các biến thể của nó để giải quyết một bài toán do nhóm tự chọn.
* Gợi ý các bài toán:
    1. Phát hiện bất thường: Phát hiện giao dịch gian lận, tìm lỗi trong dây chuyền sản xuất qua hình ảnh, phát hiện tấn công mạng.
    2. Giảm nhiễu: Loại bỏ nhiễu khỏi hình ảnh (ví dụ: ảnh mờ, ảnh cũ) hoặc tín hiệu âm thanh.
    3. Nén và Biểu diễn Dữ liệu: Nén ảnh, học các đặc trưng ẩn (latent features) của dữ liệu để phục vụ cho các tác vụ khác như phân cụm.
* Yêu cầu kỹ thuật:
    1. Chuẩn bị dữ liệu: Xử lý dữ liệu đầu vào phù hợp với mục tiêu (ví dụ: chỉ huấn luyện trên dữ liệu "bình thường" cho bài toán phát hiện bất thường; tạo các cặp dữ liệu "nhiễu-sạch" cho bài toán giảm nhiễu).
    2. Xây dựng mô hình: Bắt buộc triển khai một biến thể của Autoencoder (ví dụ: Denoising Autoencoder, Sparse Autoencoder, Variational Autoencoder - VAE).
    3. Ứng dụng và Đánh giá: Sử dụng mô hình cho mục đích đã chọn. Đánh giá kết quả dựa trên mục tiêu của bài toán (ví dụ: dùng lỗi tái tạo để tìm bất thường; dùng PSNR/SSIM để đo chất lượng ảnh sau khi giảm nhiễu).

## Yêu cầu nộp bài

* Định dạng: Một file *.ipynb duy nhất.
* Nội dung trong file Notebook: File phải được trình bày rõ ràng, mạch lạc bằng các ô Markdown và Code, bao gồm các phần sau:

    1. Giới thiệu: Mô tả bài toán, mục tiêu và bộ dữ liệu đã chọn.
    2. Tiền xử lý dữ liệu: Giải thích các bước làm sạch, chuẩn hóa, và chuẩn bị dữ liệu cho mô hình.
    3. Xây dựng mô hình: Trình bày chi tiết kiến trúc mô hình đã sử dụng (có thể vẽ sơ đồ khối bằng torchsummary hoặc các công cụ khác).
    4. Huấn luyện và Đánh giá: Mô tả quá trình huấn luyện, các tham số đã chọn. Trình bày kết quả bằng bảng, biểu đồ (loss/accuracy curves) và các độ đo phù hợp.
    5. Phân tích: Diễn giải kết quả. Mô hình hoạt động tốt/chưa tốt ở đâu? Tại sao? Có thể cải thiện như thế nào?
    6. Kết luận: Tóm tắt lại toàn bộ quá trình và kết quả đạt được.

## Tiêu chí chấm điểm (Thang điểm 10)

1. Mã nguồn và Mô hình (3.0 điểm):
    * Mã nguồn rõ ràng, có chú thích, dễ hiểu.
    * Mô hình được triển khai đúng kiến trúc, không có lỗi logic.

2. Xử lý Dữ liệu và Đánh giá (3.0 điểm):
    * Các bước tiền xử lý dữ liệu hợp lý và được giải thích rõ ràng.
    * Lựa chọn các độ đo (metrics) phù hợp với bài toán.
    * Quá trình đánh giá chặt chẽ, sử dụng tập validation/test hợp lý.

3. Báo cáo và Phân tích (4.0 điểm):
    * Cấu trúc báo cáo trong notebook rõ ràng, logic.
    * Trực quan hóa kết quả một cách hiệu quả (đồ thị, bảng biểu).
    * Phân tích sâu sắc: Không chỉ nêu ra kết quả (ví dụ: "độ chính xác là 95%"), mà cần phải diễn giải được ý nghĩa của nó, chỉ ra các hạn chế của mô hình và đề xuất hướng cải thiện.
