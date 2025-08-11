# Tổng quan về ImgAE-Dx

ImgAE-Dx là một dự án nghiên cứu tập trung vào việc phát hiện các bất thường trong ảnh X-quang ngực bằng cách sử dụng các kiến trúc Autoencoder.

Mục tiêu chính là so sánh hiệu quả của hai kiến trúc cụ thể: một kiến trúc U-Net tiêu chuẩn và một kiến trúc Reversed Autoencoder (RA) mới lạ.

## Nguyên lý hoạt động của phát hiện bất thường bằng Autoencoder

Cốt lõi của phương pháp này dựa trên ý tưởng rằng một Autoencoder được huấn luyện *chỉ trên dữ liệu "bình thường" (không có bất thường)* sẽ học được cách biểu diễn và tái tạo lại dữ liệu bình thường một cách hiệu quả.

* Khi một hình ảnh **bình thường** được đưa vào mô hình đã huấn luyện, mô hình sẽ tái tạo lại nó gần như hoàn hảo, dẫn đến **lỗi tái tạo (reconstruction error) thấp**.
* Khi một hình ảnh **bất thường** được đưa vào, mô hình sẽ gặp khó khăn trong việc tái tạo chính xác phần bất thường đó, vì nó chưa bao giờ "thấy" hoặc học cách biểu diễn bất thường. Thay vào đó, nó sẽ cố gắng tái tạo phần bất thường đó thành một dạng "bình thường" mà nó đã học được. Điều này dẫn đến **lỗi tái tạo cao** tại vị trí của bất thường.

Chính sự khác biệt về lỗi tái tạo này là cơ sở để xác định và định vị các bất thường.

## Kiến trúc mô hình (Model Architectures)

ImgAE-Dx nghiên cứu hai kiến trúc Autoencoder chính:

### 1. U-Net (Standard Autoencoder Variant)

* **Kiến trúc:** U-Net là một kiến trúc mạng nơ-ron tích chập (Convolutional Neural Network - CNN) có dạng đối xứng chữ "U", bao gồm một phần mã hóa (encoder) và một phần giải mã (decoder).
  * **Encoder:** Giống như một mạng CNN phân loại thông thường, nó bao gồm các lớp tích chập và pooling để giảm kích thước không gian của hình ảnh và trích xuất các đặc trưng cấp cao.
  * **Decoder:** Đối xứng với encoder, nó sử dụng các lớp tích chập ngược (transposed convolutions) hoặc upsampling để tăng kích thước không gian và tái tạo lại hình ảnh.
  * **Skip Connections (Kết nối bỏ qua):** Đây là đặc điểm nổi bật của U-Net. Các kết nối này truyền trực tiếp thông tin từ các lớp tương ứng trong encoder sang decoder. Điều này giúp decoder giữ lại các chi tiết không gian mịn (fine-grained details) đã bị mất trong quá trình pooling của encoder, rất quan trọng cho việc tái tạo hình ảnh chất lượng cao.
* **Vai trò trong ImgAE-Dx:** U-Net được sử dụng như một Autoencoder mạnh mẽ, có khả năng tái tạo hình ảnh rất tốt. Khi được huấn luyện trên dữ liệu bình thường, nó sẽ học cách tái tạo các cấu trúc giải phẫu bình thường một cách chi tiết. Bất kỳ sai lệch nào so với cấu trúc bình thường (tức là bất thường) sẽ dẫn đến lỗi tái tạo đáng kể.

### 2. Reversed Autoencoder (RA) (Novel Architecture)

* **Kiến trúc:** RA là một biến thể Autoencoder được thiết kế đặc biệt để khuếch đại lỗi tái tạo tại các vị trí bất thường. Dựa trên mô tả, nó có các đặc điểm sau:
  * **Asymmetric Encoder-Decoder (Mã hóa-Giải mã bất đối xứng):** Điều này có thể có nghĩa là số lượng lớp, độ sâu, hoặc dung lượng của encoder và decoder không đối xứng. Mục đích có thể là để encoder nén thông tin mạnh hơn hoặc decoder có khả năng tái tạo hạn chế hơn, buộc mô hình phải học một biểu diễn "bình thường" rất khái quát.
  * **Without Skip Connections (Không có kết nối bỏ qua):** Đây là một điểm khác biệt quan trọng so với U-Net. Việc loại bỏ các skip connections buộc thông tin phải đi qua "nút thắt cổ chai" (bottleneck) của Autoencoder. Điều này làm cho mô hình khó tái tạo lại các chi tiết nhỏ hoặc các cấu trúc phức tạp một cách hoàn hảo, đặc biệt là khi chúng không khớp với biểu diễn "bình thường" đã học.
  * **Specialized "Pseudo-Healthy" Reconstruction Design (Thiết kế tái tạo "giả khỏe mạnh" chuyên biệt):** Đây là nguyên lý cốt lõi của RA cho phát hiện bất thường. Khi mô hình RA được huấn luyện chỉ trên dữ liệu khỏe mạnh, nó sẽ học cách ánh xạ mọi đầu vào (kể cả bất thường) về một biểu diễn không gian tiềm ẩn (latent space) của "sức khỏe". Khi giải mã từ biểu diễn này, nó sẽ cố gắng tái tạo lại hình ảnh như thể nó là một cấu trúc khỏe mạnh. Do đó, nếu có một bất thường trong ảnh đầu vào, RA sẽ không thể tái tạo chính xác bất thường đó mà thay vào đó sẽ tạo ra một phiên bản "giả khỏe mạnh" của vùng đó. Sự khác biệt giữa bất thường thực tế và phiên bản "giả khỏe mạnh" này sẽ tạo ra một lỗi tái tạo rất lớn, làm nổi bật vị trí bất thường.
* **Vai trò trong ImgAE-Dx:** RA được thiết kế để tối đa hóa sự khác biệt giữa hình ảnh bất thường và bản tái tạo của nó, làm cho các bất thường trở nên rõ ràng hơn trên bản đồ lỗi tái tạo so với U-Net.

## Thuật toán phát hiện bất thường (Anomaly Detection Algorithm)

Sau khi các mô hình U-Net và RA đã được huấn luyện thành công trên tập dữ liệu hình ảnh bình thường, thuật toán phát hiện bất thường cho một hình ảnh mới sẽ diễn ra như sau:

1. **Đầu vào hình ảnh:** Một hình ảnh y khoa mới ($I_{input}$) được đưa vào mô hình Autoencoder đã huấn luyện (U-Net hoặc RA).
2. **Tái tạo hình ảnh:** Mô hình Autoencoder tạo ra một hình ảnh tái tạo ($I_{reconstructed}$).
3. **Tính toán lỗi tái tạo:** Lỗi tái tạo được tính toán bằng cách so sánh từng pixel (hoặc từng vùng) giữa $I_{input}$ và $I_{reconstructed}$. Các độ đo phổ biến bao gồm:
    * **Mean Squared Error (MSE):** $MSE = \frac{1}{N} \sum_{i=1}^{N} (I_{input,i} - I_{reconstructed,i})^2$
    * **Mean Absolute Error (MAE):** $MAE = \frac{1}{N} \sum_{i=1}^{N} |I_{input,i} - I_{reconstructed,i}|$
    * Kết quả là một bản đồ lỗi (error map) có cùng kích thước với hình ảnh đầu vào, trong đó mỗi pixel/vùng thể hiện mức độ lỗi tái tạo tại vị trí đó.
4. **Ngưỡng hóa và Định vị Bất thường:**
    * Áp dụng một ngưỡng (threshold) lên bản đồ lỗi. Các pixel/vùng có giá trị lỗi vượt quá ngưỡng được coi là bất thường. Ngưỡng này thường được xác định dựa trên phân phối lỗi tái tạo của tập dữ liệu bình thường (ví dụ: sử dụng phân vị thứ 95 hoặc 99 của lỗi trên tập validation bình thường).
    * Các vùng bất thường được khoanh vùng (ví dụ: bằng bounding box hoặc segmentation mask) để định vị chính xác vị trí của chúng trên hình ảnh gốc.
5. **Đánh giá và Trực quan hóa:**
    * Đánh giá hiệu suất phát hiện bất thường bằng các độ đo như AUC-ROC.
    * Trực quan hóa bản đồ lỗi và các vùng bất thường được khoanh vùng để bác sĩ có thể dễ dàng nhận diện.

## Điểm đặc biệt của ImgAE-Dx

Dự án ImgAE-Dx không chỉ triển khai một Autoencoder mà còn thực hiện một **nghiên cứu so sánh nghiêm ngặt** giữa U-Net và Reversed Autoencoder. Điều này cho phép đánh giá khách quan ưu nhược điểm của từng kiến trúc trong bối cảnh phát hiện bất thường không giám sát, đặc biệt là khả năng của RA trong việc khuếch đại tín hiệu bất thường thông qua cơ chế tái tạo "giả khỏe mạnh" và thiếu skip connections.
