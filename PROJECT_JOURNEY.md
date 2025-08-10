# Đề Cương Phương Pháp Luận Nghiên Cứu Dự Án

**Chủ đề:** Đánh giá so sánh hiệu quả của các kiến trúc Autoencoder trong bài toán phát hiện bất thường không giám sát trên ảnh X-quang y khoa.
**Cập nhật lần cuối:** 10/08/2025

---

## Tóm Tắt (Abstract)

Nghiên cứu này đề xuất một phương pháp luận để đánh giá và so sánh hiệu quả của hai kiến trúc mạng nơ-ron tích chập dựa trên autoencoder cho bài toán phát hiện bất thường không giám sát. Một kiến trúc U-Net tiêu chuẩn được sử dụng làm mô hình cơ sở (baseline) để đối chứng với một kiến trúc thử nghiệm, Reversed Autoencoder (RA), được lấy cảm hứng từ các nghiên cứu gần đây. Thử nghiệm sẽ được tiến hành trên một tập con của bộ dữ liệu NIH Chest X-ray. Hiệu suất của hai mô hình sẽ được đánh giá dựa trên chỉ số Diện tích dưới đường cong ROC (AUC) và phân tích định tính bản đồ lỗi tái tạo. Toàn bộ quy trình được đóng gói trong một Jupyter Notebook có thể tái lập trên Google Colab, tích hợp cơ chế checkpointing để xử lý các phiên làm việc dài.

---

## 1. Giới Thiệu & Phát Biểu Vấn Đề (Introduction & Problem Statement)

Phát hiện bất thường trong ảnh y khoa là một nhiệm vụ quan trọng nhưng đầy thách thức, thường bị giới hạn bởi sự khan hiếm của dữ liệu được gán nhãn chi tiết. Các phương pháp học không giám sát, đặc biệt là các mô hình dựa trên autoencoder, mang lại một hướng tiếp cận tiềm năng bằng cách chỉ học từ dữ liệu "bình thường". Tuy nhiên, việc lựa chọn kiến trúc autoencoder phù hợp có ảnh hưởng lớn đến hiệu suất. Nghiên cứu này giải quyết vấn đề đó bằng cách so sánh một kiến trúc tiêu chuẩn (U-Net) với một kiến trúc mới nổi (RA) trong một bối cảnh có kiểm soát.

---

## 2. Câu Hỏi Nghiên Cứu & Giả Thuyết (Research Questions & Hypotheses)

- **RQ1:** Kiến trúc U-Net, với các kết nối tắt, có khả năng thiết lập một baseline hiệu quả (AUC > 0.80) cho việc phân loại ảnh X-quang bình thường và bất thường dựa trên lỗi tái tạo không?
- **RQ2:** Kiến trúc Reversed Autoencoder, được thiết kế để tái tạo ảnh "giả lành tính", có cho thấy hiệu suất vượt trội hơn (về mặt AUC và/hoặc chất lượng bản đồ lỗi) so với kiến trúc U-Net tiêu chuẩn không?

- **H1 (Giả thuyết 1):** Mô hình U-Net sẽ đạt được hiệu suất baseline tốt, có khả năng phân biệt rõ ràng giữa hai lớp dữ liệu.
- **H2 (Giả thuyết 2):** Mô hình RA sẽ tạo ra các bản đồ lỗi có tính khu trú cao hơn và có thể đạt được chỉ số AUC cao hơn U-Net do cơ chế tái tạo chuyên biệt của nó.

---

## 3. Phương Pháp Luận (Methodology)

### 3.1. Thiết Kế Nghiên Cứu

Nghiên cứu này sử dụng **thiết kế thực nghiệm so sánh (comparative experimental design)**. Hai mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu và quy trình để đảm bảo tính công bằng và cho phép rút ra kết luận về hiệu quả tương đối của chúng.

### 3.2. Thu Thập và Chuẩn Bị Dữ Liệu

- **Nguồn dữ liệu:** Bộ dữ liệu **NIH Chest X-ray**.
- **Lấy mẫu:** Một tập con gồm ~2500 ảnh "bình thường" (No Finding) và ~1200 ảnh "bất thường" sẽ được trích xuất để đảm bảo thử nghiệm có thể hoàn thành trong thời gian hợp lý.
- **Phân chia dữ liệu:** Dữ liệu "bình thường" được chia thành tập huấn luyện (train) và tập xác thực (validation). Tập kiểm tra (test) bao gồm cả dữ liệu "bình thường" và "bất thường" chưa từng thấy.
- **Tiền xử lý:** Tất cả các ảnh được chuyển đổi sang ảnh xám, resize về kích thước 128x128 pixels, và chuẩn hóa giá trị pixel về khoảng [-1, 1].

### 3.3. Công Cụ (Instrumentation)

- **Mô hình Baseline:** **U-Net**, một mạng mã hóa-giải mã đối xứng với các kết nối tắt để bảo toàn thông tin không gian.
- **Mô hình Thử nghiệm:** **Reversed Autoencoder (RA)**, một mạng mã hóa-giải mã không đối xứng, không có kết nối tắt, buộc việc tái tạo phải dựa chủ yếu vào thông tin đã được nén ở không gian ẩn.

### 3.4. Quy Trình Thử Nghiệm

1. Cả hai mô hình được huấn luyện độc lập trên cùng một tập huấn luyện chỉ chứa dữ liệu "bình thường".
2. Hàm mất mát **Mean Squared Error (MSE)** và trình tối ưu hóa **Adam** được sử dụng cho cả hai.
3. Mô hình có loss thấp nhất trên tập xác thực sẽ được lưu lại.
4. Các mô hình tốt nhất sau đó được đánh giá trên tập kiểm tra.
5. **Cơ chế Checkpointing:** Một cơ chế checkpointing được tích hợp vào vòng lặp huấn luyện. Sau mỗi epoch, toàn bộ trạng thái huấn luyện (trọng số model, trạng thái optimizer, epoch hiện tại, lịch sử loss) sẽ được lưu vào một bộ nhớ bền vững. Điều này đảm bảo quá trình huấn luyện có thể được khôi phục liền mạch nếu phiên làm việc bị gián đoạn.

### 3.5. Các Chỉ Số Đánh Giá

- **Định lượng:** **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** là chỉ số chính để đo lường khả năng phân loại tổng thể.
- **Định tính:** **Trực quan hóa bản đồ lỗi tái tạo (Reconstruction Error Maps)** để phân tích khả năng khu trú bất thường của từng mô hình.

---

## 4. Chiến Lược Triển Khai

- **Nền tảng:** **Google Colab** được chọn để đảm bảo khả năng tái lập và tận dụng GPU miễn phí.
- **Cấu trúc:** Dự án được triển khai trong một **Jupyter Notebook duy nhất có cấu trúc module hóa** (`Anomaly_Detection_Research_Colab.ipynb`).
- **Lưu trữ bền vững:** **Google Drive** được tích hợp vào quy trình làm việc. Các file checkpoint được tự động lưu vào một thư mục được chỉ định trên Google Drive của người dùng để chống lại việc mất dữ liệu do giới hạn thời gian của phiên Colab.

---

## 5. Tình Trạng & Kế Hoạch Thực Thi

- **Tình trạng hiện tại:** Toàn bộ đề cương nghiên cứu và phương pháp luận đã được xác định. File Notebook `Anomaly_Detection_Research_Colab.ipynb` chứa mã nguồn triển khai phương pháp luận này đã được tạo.
- **Kế hoạch thực thi:**
    1. Mở Notebook trên Google Colab và kết nối với Google Drive.
    2. Thực thi **Phần 0** để thiết lập môi trường và tải dữ liệu (yêu cầu có file `kaggle.json`).
    3. Thực thi tuần tự các phần còn lại. Quá trình huấn luyện sẽ **tự động tìm và tải checkpoint** nếu có, hoặc bắt đầu từ đầu nếu không.
    4. Nếu phiên làm việc bị ngắt, người dùng chỉ cần thực thi lại các ô từ đầu. Quá trình huấn luyện sẽ tự động tiếp tục từ nơi nó đã dừng.

---

## 6. Các Hạn Chế & Hướng Phát Triển Tương Lai

- **Hạn chế của nghiên cứu:**
  - Việc sử dụng tập dữ liệu con có thể không phản ánh đầy đủ hiệu suất trên toàn bộ dữ liệu.
  - Số lượng epoch huấn luyện bị giới hạn có thể khiến các mô hình chưa hội tụ hoàn toàn.
  - Nghiên cứu chỉ so sánh hai kiến trúc và một hàm loss duy nhất.
- **Hướng phát triển tương lai:**
  - Mở rộng thử nghiệm trên toàn bộ bộ dữ liệu.
  - Thực hiện tinh chỉnh siêu tham số một cách có hệ thống.
  - So sánh các hàm loss khác nhau (ví dụ: SSIM, L1).
  - Kiểm tra khả năng tổng quát hóa của mô hình tốt nhất trên các bộ dữ liệu khác (MURA, Brain MRI).
