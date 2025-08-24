# Đề cương phương pháp luận nghiên cứu dự án

**Chủ đề:** Đánh giá so sánh hiệu quả của các kiến trúc Autoencoder trong bài toán phát hiện bất thường không giám sát trên ảnh X-quang y khoa.
**Cập nhật lần cuối:** 24/08/2025

---

## Tóm tắt (abstract)

Nghiên cứu này đề xuất một phương pháp luận để đánh giá và so sánh hiệu quả của hai kiến trúc mạng nơ-ron tích chập dựa trên autoencoder cho bài toán phát hiện bất thường không giám sát. Một kiến trúc U-Net tiêu chuẩn được sử dụng làm mô hình cơ sở (baseline) để đối chứng với một kiến trúc thử nghiệm, Reversed Autoencoder (RA), được lấy cảm hứng từ các nghiên cứu gần đây. Thử nghiệm sẽ được tiến hành trên bộ dữ liệu Chest X-ray Pneumonia từ HuggingFace. Hiệu suất của hai mô hình sẽ được đánh giá dựa trên chỉ số Diện tích dưới đường cong ROC (AUC) và phân tích định tính bản đồ lỗi tái tạo. Toàn bộ quy trình được đóng gói trong một Jupyter Notebook có thể tái lập trên Google Colab, tích hợp cơ chế checkpointing để xử lý các phiên làm việc dài.

---

## 1. Giới thiệu & phát biểu vấn đề (introduction & problem statement)

Phát hiện bất thường trong ảnh y khoa là một nhiệm vụ quan trọng nhưng đầy thách thức, thường bị giới hạn bởi sự khan hiếm của dữ liệu được gán nhãn chi tiết. Các phương pháp học không giám sát, đặc biệt là các mô hình dựa trên autoencoder, mang lại một hướng tiếp cận tiềm năng bằng cách chỉ học từ dữ liệu "bình thường". Tuy nhiên, việc lựa chọn kiến trúc autoencoder phù hợp có ảnh hưởng lớn đến hiệu suất. Nghiên cứu này giải quyết vấn đề đó bằng cách so sánh một kiến trúc tiêu chuẩn (U-Net) với một kiến trúc mới nổi (RA) trong một bối cảnh có kiểm soát.

---

## 2. Câu hỏi nghiên cứu & giả thuyết (research questions & hypotheses)

- **RQ1: U-Net architecture and anomaly detection baseline**
  - Kiến trúc U-Net, với các kết nối tắt, có khả năng thiết lập một baseline hiệu quả (AUC > 0.80) cho việc phân loại ảnh X-quang bình thường (NORMAL) và bất thường (PNEUMONIA) dựa trên lỗi tái tạo không?
  - Can the U-Net architecture, leveraging its **skip connections**, establish an effective baseline (AUC > 0.80) for classifying normal versus pneumonia X-ray images based on **reconstruction error**?

- **RQ2: Reversed Autoencoder vs U-Net performance**
  - Kiến trúc Reversed Autoencoder, được thiết kế để tái tạo ảnh "giả lành tính", có cho thấy hiệu suất vượt trội hơn (về mặt AUC và (hoặc) chất lượng bản đồ lỗi) so với kiến trúc U-Net tiêu chuẩn không?
  - Does the **Reversed Autoencoder (RA)** architecture, specifically designed to reconstruct images into a "**pseudo-healthy**" state, demonstrate superior performance (in terms of AUC and/or the quality of the **error map**) compared to the standard U-Net architecture?

- **H1: U-Net's effectiveness as a baseline**
  - Mô hình U-Net sẽ đạt được hiệu suất baseline tốt (AUC > 0.80), có khả năng phân biệt rõ ràng giữa ảnh X-quang bình thường và ảnh có viêm phổi.
  - The U-Net model will achieve good baseline performance (AUC > 0.80), demonstrating a clear discriminative ability between normal and pneumonia cases.

- **H2: Specialized performance of the reversed autoencoder**
  - Mô hình RA sẽ tạo ra các bản đồ lỗi có tính khu trú cao hơn tại vùng viêm phổi, mặc dù AUC có thể tương đương hoặc thấp hơn một chút so với U-Net do thiết kế chuyên biệt của nó.
  - The RA model will produce **more localized error maps** at pneumonia regions, though AUC may be comparable or slightly lower than U-Net due to its specialized reconstruction mechanism.

---

## 3. Phương Pháp Luận (Methodology)

### 3.1. Thiết Kế Nghiên Cứu

Nghiên cứu này sử dụng **thiết kế thực nghiệm so sánh (comparative experimental design)**. Hai mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu và quy trình để đảm bảo tính công bằng và cho phép rút ra kết luận về hiệu quả tương đối của chúng.

### 3.2. Thu Thập và Chuẩn Bị Dữ Liệu

- **Nguồn dữ liệu:** Bộ dữ liệu **HuggingFace Chest X-ray Pneumonia** (`hf-vision/chest-xray-pneumonia`).
- **Đặc điểm dữ liệu:** 
  - 2 classes: NORMAL (label 0) và PNEUMONIA (label 1)
  - Format: Parquet files, tương thích với HuggingFace datasets mới nhất
  - Dữ liệu đã được chuẩn hóa và sẵn sàng sử dụng
- **Lấy mẫu:** 
  - Training: ~2000-3000 ảnh NORMAL (chỉ dùng ảnh bình thường)
  - Validation: ~500 ảnh NORMAL
  - Test: ~500 ảnh NORMAL + ~500 ảnh PNEUMONIA (cân bằng)
- **Tiền xử lý:** 
  - Resize về kích thước 128x128 pixels
  - Chuyển sang grayscale (1 channel)
  - Chuẩn hóa giá trị pixel về khoảng [-1, 1]

### 3.3. Công Cụ (Instrumentation)

- **Mô hình Baseline:** **U-Net** (~55M parameters)
  - Encoder-decoder đối xứng với skip connections
  - Bottleneck: 8×8 spatial dimension
  - Single-layer FC mapping
  
- **Mô hình Thử nghiệm:** **Reversed Autoencoder (RA)** (~273M parameters)
  - Encoder-decoder bất đối xứng, không có skip connections
  - Bottleneck: 16×16 spatial dimension (lớn hơn để bù thông tin)
  - Multi-layer FC với hidden layers

### 3.4. Quy Trình Thử Nghiệm

1. **Data Loading:** Sử dụng HuggingFace datasets API với streaming option
2. **Training Protocol:**
   - Huấn luyện độc lập trên tập NORMAL images
   - Loss function: Mean Squared Error (MSE)
   - Optimizer: Adam với learning rate 1e-4
   - Batch size: 48 (T4 GPU optimized)
   - Epochs: 20-30 với early stopping
3. **Checkpointing:** 
   - Lưu model state mỗi 5 epochs
   - Best model based on validation loss
   - Google Drive integration cho persistence
4. **Mixed Precision Training:** Sử dụng PyTorch AMP cho T4 GPU efficiency

### 3.5. Các Chỉ Số Đánh Giá

- **Định lượng:** 
  - **AUC-ROC:** Khả năng phân loại tổng thể (primary metric)
  - **AUC-PR:** Precision-Recall curve
  - **F1-Score:** Balance precision/recall
  - **Sensitivity/Specificity:** Medical relevance
  
- **Định tính:** 
  - **Error Heatmaps:** Visualization của reconstruction error
  - **Localization Quality:** Đánh giá khả năng định vị vùng viêm phổi
  - **Side-by-side Comparison:** Original | Reconstructed | Error Map

---

## 4. Chiến Lược Triển Khai

- **Nền tảng:** **Google Colab** với T4 GPU
- **Notebook:** `ImgAE_Dx_HuggingFace_Training_Fixed.ipynb`
- **Key Features:**
  - HuggingFace integration (no local data download needed)
  - Automatic GPU detection và optimization
  - Google Drive checkpointing
  - W&B experiment tracking (optional)

---

## 5. Tình Trạng & Kế Hoạch Thực Thi

### 5.1. Tình Trạng Hiện Tại (24/08/2025)

✅ **Đã hoàn thành:**
- Framework code hoàn chỉnh với 63 tests (>90% coverage)
- Both models implemented và validated
- HuggingFace dataset integration working
- Notebook ready cho Google Colab T4 GPU
- Fixed PyTorch AMP deprecation warnings

⏳ **Đang chờ thực hiện:**
- Full training run với 20-30 epochs
- Evaluation trên test set (NORMAL + PNEUMONIA)
- Statistical comparison của 2 models
- Research questions validation

### 5.2. Kế Hoạch Thực Thi

1. **Setup Phase:**
   - Upload notebook lên Google Colab
   - Enable T4 GPU runtime
   - Mount Google Drive cho checkpoints

2. **Training Phase:**
   ```python
   # Expected timeline
   - U-Net: ~20-30 minutes (20 epochs)
   - Reversed AE: ~40-60 minutes (20 epochs)
   - Total: ~1-2 hours với checkpointing
   ```

3. **Evaluation Phase:**
   - Load best checkpoints
   - Compute reconstruction errors trên test set
   - Calculate AUC-ROC cho NORMAL vs PNEUMONIA
   - Generate error heatmaps

4. **Analysis Phase:**
   - Compare AUC scores
   - Analyze error localization quality
   - Statistical significance testing
   - Document findings

---

## 6. Expected Results & Research Validation

### 6.1. Quantitative Expectations

| Metric | U-Net (Expected) | Reversed AE (Expected) |
|--------|------------------|------------------------|
| AUC-ROC | 0.85-0.90 | 0.80-0.85 |
| Training Loss | < 0.01 | < 0.02 |
| Inference Time | ~10ms/image | ~15ms/image |

### 6.2. Qualitative Expectations

- **U-Net:** Clear reconstruction với sharp details, error concentrated at pneumonia regions
- **RA:** "Pseudo-healthy" reconstruction, potentially stronger error signal at anomalies

### 6.3. Research Questions Validation

**Để trả lời RQ1:** 
- Nếu U-Net AUC > 0.80 → Baseline effectiveness confirmed ✓
- Analyze ROC curve và optimal threshold

**Để trả lời RQ2:**
- Compare AUC scores với statistical test
- Visual comparison của error maps quality
- Localization accuracy at pneumonia regions

---

## 7. Các Hạn Chế & Hướng Phát Triển

### 7.1. Hạn chế

- **Dataset:** Binary classification only (NORMAL/PNEUMONIA)
- **Image Size:** Fixed 128×128 (computational constraint)
- **Single Modality:** Chest X-ray only
- **Limited Epochs:** 20-30 epochs due to time constraints

### 7.2. Hướng Phát Triển

- **Multi-class:** Extend to multiple pathologies
- **Higher Resolution:** 256×256 or 512×512 images
- **Cross-modality:** Test on CT, MRI datasets
- **Advanced Architectures:** Vision Transformers, Attention mechanisms
- **Ensemble Methods:** Combine U-Net + RA predictions

---

## 8. Conclusion

Nghiên cứu này sẽ cung cấp evidence-based comparison giữa U-Net và Reversed Autoencoder cho medical anomaly detection. Với HuggingFace dataset integration và Google Colab setup, methodology này có thể được reproduce và extend bởi research community.

**Next Steps:** Execute training protocol và collect results để validate research hypotheses.