# Kế Hoạch Thực Thi & Cơ Sở Lý Luận Chi Tiết

**Mục đích:** File này là một tài liệu tổng hợp, vừa là checklist các bước cần thực hiện trong file `Anomaly_Detection_Research_Colab.ipynb`, vừa cung cấp kiến thức và lý do (rationale) đằng sau mỗi giai đoạn theo phương pháp luận nghiên cứu đã đề ra.

---

## **Giai đoạn 0: Thiết Lập Môi Trường & Dữ Liệu**

### **Checklist Công Việc:**

- [ ] Chạy ô kết nối Google Drive.
- [ ] Chạy ô cài đặt các thư viện Python cần thiết.
- [ ] Chạy ô tải lên file `kaggle.json` để xác thực API.
- [ ] Chạy ô tải và giải nén bộ dữ liệu NIH Chest X-ray.

### **Cơ Sở Lý Luận & Kiến Thức Chuyên Sâu:**

- **Mục đích:** Giai đoạn này nhằm thiết lập một môi trường làm việc có khả năng tái lập (reproducible) và bền vững (persistent).
- **Tại sao cần Môi trường ảo/Colab?:** Chúng ta cô lập các thư viện của dự án để tránh xung đột phiên bản ("dependency hell"), đảm bảo rằng bất kỳ ai cũng có thể chạy lại thí nghiệm này với cùng một bộ công cụ.
- **Tại sao dùng Kaggle API?:** Thay vì tải thủ công các file dữ liệu lớn, việc dùng API đảm bảo tính tự động hóa, tốc độ, và quan trọng nhất là tính nhất quán. Mọi lần chạy đều sẽ lấy về cùng một phiên bản dữ liệu gốc.
- **Tại sao phải kết nối Google Drive?:** Đây là bước tối quan trọng để giải quyết bài toán giới hạn của Colab. Các phiên Colab là tạm thời và sẽ bị xóa sau một thời gian. Bằng cách kết nối Google Drive, chúng ta có một bộ nhớ bền vững. Đây là nền tảng cho kỹ thuật **checkpointing** - một phương pháp tiêu chuẩn trong các dự án học sâu dài hơi, giúp lưu lại tiến trình và phục hồi sau khi bị ngắt kết nối.

---

## **Giai đoạn 1-3: Cấu Hình, Định Nghĩa & Chuẩn Bị Dữ Liệu**

### **Checklist Công Việc:**

- [ ] Chạy ô **Phần 1** để thiết lập các biến cấu hình toàn cục.
- [ ] Chạy ô **Phần 2** để khai báo lớp (class) cho hai kiến trúc `UNet` và `ReversedAutoencoder`.
- [ ] Chạy ô **Phần 3** để thực thi logic xử lý và tạo ra các `DataLoader`.

### **Cơ Sở Lý Luận & Kiến Thức Chuyên Sâu:**

- **Mục đích:** Các giai đoạn này định hình nên "bộ khung" và các quy tắc của thử nghiệm, đảm bảo một sự so sánh công bằng.
- **Kiến thức về Cấu hình:** Việc tập trung các siêu tham số (learning rate, batch size,...) vào một nơi duy nhất hoạt động như một "bảng điều khiển" cho thí nghiệm, cho phép chúng ta dễ dàng lặp lại các thử nghiệm với các thiết lập khác nhau mà không cần thay đổi code logic.
- **Kiến thức về Thiết kế Mô hình:** Cốt lõi của nghiên cứu này là một cuộc "đối đầu" giữa hai triết lý kiến trúc:
  - **U-Net (Baseline):** Được chọn vì nó là "tiêu chuẩn vàng" trong phân khúc ảnh y khoa. Các **kết nối tắt (skip connections)** của nó được thiết kế để bảo toàn thông tin không gian ở nhiều cấp độ phân giải, giúp nó tái tạo hình ảnh cực kỳ chi tiết. Đây là một "đối thủ" rất mạnh.
  - **Reversed Autoencoder (Thử nghiệm):** Chúng ta cố tình xây dựng một kiến trúc không đối xứng và loại bỏ các kết nối tắt. Lý thuyết đằng sau là việc ép toàn bộ thông tin phải đi qua một "cổ chai" (bottleneck) thật hẹp sẽ buộc mô hình phải học những đặc trưng cốt lõi nhất của dữ liệu "bình thường" và phải "vứt bỏ" những chi tiết không quan trọng, trong đó có thể bao gồm cả các bất thường mà nó chưa từng thấy.
- **Kiến thức về Đóng gói Dữ liệu:** Lớp `Dataset` và `DataLoader` của PyTorch là những abstraction cơ bản nhưng cực kỳ mạnh mẽ. Chúng giúp tách biệt hoàn toàn logic chuẩn bị dữ liệu (đọc file, resize, chuẩn hóa) khỏi logic huấn luyện mô hình. Điều này không chỉ giúp code sạch hơn mà còn đảm bảo cả hai mô hình đều nhận được đầu vào y hệt nhau, một điều kiện tiên quyết cho một so sánh khoa học hợp lệ.

---

## **Giai đoạn 4-5: Huấn Luyện & Theo Dõi**

### **Checklist Công Việc:**

- [ ] Chạy ô **Phần 4** để định nghĩa hàm `train_model_with_checkpointing`.
- [ ] Chạy ô **Phần 5a** để huấn luyện mô hình U-Net.
- [ ] Chạy ô **Phần 5b** để huấn luyện mô hình Reversed Autoencoder.
- [ ] Chạy ô **Phần 5c** để vẽ biểu đồ loss.

### **Cơ Sở Lý Luận & Kiến Thức Chuyên Sâu:**

- **Mục đích:** Thực thi thí nghiệm và thu thập dữ liệu về quá trình học của hai mô hình.
- **Kiến thức về Paradim Huấn luyện:** Chúng ta đang áp dụng mô hình **phát hiện bất thường không giám sát (unsupervised anomaly detection)**. Bằng cách chỉ cho mô hình xem ảnh "bình thường", chúng ta đang dạy cho nó "khái niệm về sự bình thường". Giả thuyết khoa học ở đây là: khi đối mặt với một hình ảnh bất thường (nằm ngoài phân phối dữ liệu mà nó đã học), khả năng tái tạo của mô hình sẽ kém đi đáng kể, dẫn đến **lỗi tái tạo (reconstruction error)** cao.
- **Kiến thức về Checkpointing:** Hàm `train_model_with_checkpointing` là "người hùng thầm lặng" của dự án. Nó không chỉ lưu trọng số mô hình, mà lưu cả trạng thái của optimizer, epoch hiện tại, và lịch sử loss. Việc này đảm bảo rằng khi tiếp tục huấn luyện, quá trình học sẽ liền mạch như chưa từng bị gián đoạn, kể cả learning rate và momentum của Adam optimizer.

---

## **Giai đoạn 6: Đánh Giá & Phân Tích So Sánh**

### **Checklist Công Việc:**

- [ ] Chạy ô **6a** để định nghĩa các hàm hỗ trợ đánh giá.
- [ ] Chạy ô **6b** để tải các model tốt nhất và tính toán chỉ số AUC.
- [ ] Chạy ô **6c** để trực quan hóa đường cong ROC và các hình ảnh so sánh.

### **Cơ Sở Lý Luận & Kiến Thức Chuyên Sâu:**

- **Mục đích:** Chuyển đổi dữ liệu thô (lỗi tái tạo) thành các bằng chứng có thể diễn giải được để trả lời câu hỏi nghiên cứu.
- **Kiến thức về Đánh giá:** Việc đánh giá được tiến hành trên hai phương diện:
    1. **Phân tích Định lượng (Quantitative):** Chúng ta dùng **AUC-ROC** vì nó là một chỉ số toàn diện, không phụ thuộc vào một ngưỡng (threshold) cụ thể nào. Nó trả lời câu hỏi: "Nhìn chung, mô hình có khả năng phân biệt giữa hai nhóm tốt đến mức nào?".
    2. **Phân tích Định tính (Qualitative):** Chúng ta phân tích **bản đồ lỗi tái tạo**. Đây là bước quan trọng để tiến gần hơn đến ứng dụng lâm sàng, vì nó trả lời câu hỏi: "Mô hình nghĩ rằng vùng bất thường nằm ở đâu?". Một mô hình tốt không chỉ cho AUC cao, mà còn phải tạo ra các "điểm nóng" trên bản đồ lỗi trùng khớp với vị trí bệnh lý thực tế.

---

## **Giai đoạn 7: Kết Luận Nghiên Cứu**

### **Checklist Công Việc:**

- [ ] Dựa vào các kết quả ở Phần 6, điền các phân tích và kết luận của bạn vào ô Markdown cuối cùng.

### **Cơ Sở Lý Luận & Kiến Thức Chuyên Sâu:**

- **Mục đích:** Tổng hợp các bằng chứng và trình bày một lập luận khoa học.
- **Kiến thức:** Đây là phần thể hiện tư duy phản biện. Một kết luận tốt không chỉ nói "mô hình A tốt hơn mô hình B". Nó cần phải:
  - **Trả lời câu hỏi nghiên cứu:** Dựa trên AUC và bản đồ lỗi, mô hình nào thực sự vượt trội?
  - **Giải thích "Tại sao?":** Cố gắng liên kết kết quả quan sát được với sự khác biệt về kiến trúc. (Ví dụ: "Hiệu suất vượt trội của U-Net có thể được lý giải bởi các kết nối tắt, giúp nó...").
  - **Thừa nhận Hạn chế:** Một phần không thể thiếu của bất kỳ nghiên cứu nào là chỉ ra những yếu tố có thể ảnh hưởng đến kết quả (ví dụ: tập dữ liệu nhỏ, số epoch ít). Điều này thể hiện sự trung thực và hiểu biết sâu sắc.
  - **Đề xuất Hướng đi Mới:** Dựa trên những gì đã học được, đề xuất các thí nghiệm tiếp theo để cải thiện kết quả.
