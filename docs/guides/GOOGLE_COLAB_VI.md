# 🚀 Hướng dẫn chạy ImgAE-Dx trên Google Colab

**Hướng dẫn chi tiết để chạy dự án phát hiện bất thường trong ảnh X-quang y tế trên Google Colab với GPU T4**

---

## 📋 Tổng quan

ImgAE-Dx là dự án nghiên cứu so sánh hiệu quả của các kiến trúc Autoencoder (U-Net vs Reversed Autoencoder) trong việc phát hiện bất thường trên ảnh X-quang y tế. Dự án được thiết kế để chạy hoàn toàn trên Google Colab với GPU T4 miễn phí.

### 🎯 Những gì bạn sẽ có sau khi hoàn thành:
- ✅ Hai mô hình AI đã được huấn luyện (U-Net và Reversed Autoencoder)
- ✅ Kết quả so sánh hiệu suất phát hiện bất thường 
- ✅ Biểu đồ và phân tích chất lượng mô hình
- ✅ Checkpoint mô hình được lưu trên Google Drive

---

## 🚀 Bước 1: Chuẩn bị Google Colab

### 1.1 Truy cập Google Colab
1. Mở trình duyệt và đi tới [Google Colab](https://colab.research.google.com/)
2. Đăng nhập bằng tài khoản Google của bạn

### 1.2 Upload Notebook
1. **Tải notebook từ GitHub:**
   - Vào [Repository ImgAE-Dx](https://github.com/kinhluan/ImgAE-Dx)
   - Mở thư mục `notebooks/`
   - Click vào file `T4_GPU_Training_Colab.ipynb`
   - Click nút **"Download"** hoặc **"Raw"** rồi Save As

2. **Upload lên Colab:**
   - Trong Google Colab, click **"File" → "Upload notebook"**
   - Chọn file `T4_GPU_Training_Colab.ipynb` vừa tải về
   - Hoặc dùng link trực tiếp: `https://colab.research.google.com/github/kinhluan/ImgAE-Dx/blob/main/notebooks/T4_GPU_Training_Colab.ipynb`

### 1.3 Cấu hình GPU T4
1. **Chọn runtime GPU:**
   - Click **"Runtime" → "Change runtime type"**
   - **Hardware accelerator**: Chọn **"T4 GPU"**
   - **Runtime shape**: Chọn **"Standard"** 
   - Click **"Save"**

2. **Kiểm tra GPU:**
   ```python
   # Chạy trong cell để kiểm tra
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

---

## 🗂️ Bước 2: Thiết lập Google Drive

### 2.1 Kết nối Google Drive
Khi chạy notebook, bạn sẽ được yêu cầu kết nối Google Drive để:
- ✅ Lưu trữ mô hình đã huấn luyện
- ✅ Backup kết quả thử nghiệm
- ✅ Tiếp tục huấn luyện sau khi bị ngắt kết nối

**Các bước:**
1. Chạy cell đầu tiên trong notebook
2. Click vào link xác thực hiện ra
3. Chọn tài khoản Google và cho phép truy cập
4. Copy mã xác thực và paste vào notebook

### 2.2 Cấu trúc thư mục được tạo
```
📁 Google Drive/MyDrive/
├── 📁 imgae_dx_checkpoints/    # Mô hình đã huấn luyện
├── 📁 imgae_dx_configs/        # Cấu hình T4
├── 📁 imgae_dx_logs/          # Logs huấn luyện
└── 📁 imgae_dx_results/       # Kết quả và biểu đồ
```

---

## 🏋️ Bước 3: Cấu hình huấn luyện

### 3.1 Tham số cơ bản (có thể điều chỉnh)
```python
config = {
    'model_type': 'both',        # 'unet', 'reversed_ae', hoặc 'both'
    'samples': 3000,             # Số lượng mẫu huấn luyện  
    'epochs': 20,                # Số epochs
    'batch_size': 48,            # Kích thước batch (T4 tối ưu)
}
```

### 3.2 Các chế độ huấn luyện

#### 🧪 **Chế độ thử nghiệm nhanh** (15-20 phút)
```python
config = {
    'model_type': 'unet',
    'samples': 1500,
    'epochs': 10,
}
```

#### ⚡ **Chế độ tiêu chuẩn** (45-60 phút)  
```python
config = {
    'model_type': 'both',
    'samples': 3000,
    'epochs': 20,
}
```

#### 🔬 **Chế độ nghiên cứu** (75-90 phút)
```python
config = {
    'model_type': 'both', 
    'samples': 5000,
    'epochs': 30,
}
```

---

## ▶️ Bước 4: Chạy huấn luyện

### 4.1 Chạy tuần tự các cell
1. **Cell 1-2**: Thiết lập môi trường và Google Drive
2. **Cell 3-4**: Cài đặt dependencies và download dữ liệu
3. **Cell 5-6**: Cấu hình T4 và khởi tạo mô hình
4. **Cell 7-8**: Bắt đầu huấn luyện
5. **Cell 9-10**: Đánh giá và so sánh kết quả

### 4.2 Theo dõi quá trình huấn luyện

#### 📊 **Weights & Biases Dashboard**
- Notebook sẽ cung cấp link đến W&B dashboard
- Theo dõi real-time: loss, accuracy, GPU usage
- So sánh hiệu suất 2 mô hình

#### 🖥️ **Console Logs**
```
🚀 Starting T4-Optimized Training
==============================
✅ T4 GPU detected: Tesla T4 (16GB)
✅ Mixed precision enabled
✅ Training samples: 3000
✅ Models: U-Net + Reversed Autoencoder

📊 Epoch 1/20:
  • U-Net Loss: 0.0234 | Time: 45s
  • Rev-AE Loss: 0.0198 | Time: 47s
  
📊 Epoch 2/20:
  • U-Net Loss: 0.0189 | Time: 44s  
  • Rev-AE Loss: 0.0156 | Time: 46s
...
```

### 4.3 Lưu tự động (Checkpointing)
- ✅ Mô hình được lưu **mỗi 2 epochs**
- ✅ Backup tự động lên **Google Drive**
- ✅ Có thể **tiếp tục** sau khi bị ngắt kết nối

---

## 📊 Bước 5: Xem kết quả

### 5.1 Kết quả so sánh tự động
```
🏆 Final Comparison Results
===========================
📈 U-Net Performance:
  • AUC-ROC: 0.847
  • Training time: 23.4 minutes
  • Memory usage: 13.2GB/16GB

📈 Reversed Autoencoder Performance:  
  • AUC-ROC: 0.863
  • Training time: 24.1 minutes
  • Memory usage: 12.8GB/16GB

🎯 Winner: Reversed Autoencoder (+1.6% AUC)
```

### 5.2 Biểu đồ và phân tích
- **ROC Curves**: So sánh khả năng phân loại
- **Error Maps**: Bản đồ lỗi tái tạo trên ảnh thử nghiệm
- **Training Curves**: Quá trình học của mô hình
- **Sample Predictions**: Ví dụ phát hiện bất thường

### 5.3 Tải về kết quả
```python
# Tải toàn bộ kết quả về máy
from google.colab import files
import zipfile

# Nén kết quả
!zip -r imgae_dx_results.zip /content/drive/MyDrive/imgae_dx_results/
files.download('imgae_dx_results.zip')
```

---

## ⚠️ Xử lý sự cố

### 🚨 Lỗi thường gặp

#### **1. Lỗi hết bộ nhớ GPU**
```
RuntimeError: CUDA out of memory
```
**Giải pháp:**
```python
# Giảm batch size trong config
config['batch_size'] = 32  # từ 48 xuống 32
# Hoặc 
config['batch_size'] = 16  # nếu vẫn lỗi
```

#### **2. Colab bị ngắt kết nối**
```
Your session crashed after timing out.
```
**Giải pháp:**
- ✅ Kết nối lại và chạy từ cell cuối cùng
- ✅ Mô hình đã được lưu tự động
- ✅ Có thể tiếp tục từ checkpoint

#### **3. Lỗi tải dữ liệu**
```
DatasetError: Failed to download dataset
```
**Giải pháp:**
```python
# Thử dataset khác trong config
config['hf_dataset'] = 'Francesco/chest-xray-pneumonia-detection'
```

#### **4. Không có GPU T4**
```
GPU T4 not available
```
**Giải pháp:**
- Đợi vài phút và thử lại
- Hoặc dùng **Colab Pro** cho GPU ổn định hơn

### 💡 Tips tối ưu hóa

#### **Tăng tốc độ huấn luyện:**
```python
config.update({
    'mixed_precision': True,     # Đã bật mặc định
    'num_workers': 2,           # Tải dữ liệu song song  
    'pin_memory': True,         # Tăng tốc GPU transfer
})
```

#### **Tiết kiệm bộ nhớ:**
```python
config.update({
    'memory_limit_gb': 12,      # Giới hạn VRAM
    'batch_size': 32,           # Batch nhỏ hơn
    'image_size': 96,           # Ảnh nhỏ hơn (thay vì 128)
})
```

---

## 🔬 Hiểu về kết quả nghiên cứu

### 📈 Chỉ số AUC-ROC
- **0.5**: Không tốt hơn random
- **0.7-0.8**: Hiệu suất khá tốt  
- **0.8-0.9**: Hiệu suất tốt
- **0.9+**: Hiệu suất xuất sắc

### 🎯 Ý nghĩa so sánh
- **U-Net**: Mô hình chuẩn với skip connections
- **Reversed Autoencoder**: Kiến trúc thử nghiệm cho phát hiện bất thường
- **Mục tiêu**: Tìm ra kiến trúc tốt nhất cho ảnh y tế

### 📋 Ứng dụng thực tế
- Hỗ trợ bác sĩ phát hiện bệnh lý trên X-quang
- Sàng lọc tự động ảnh bất thường
- Nghiên cứu AI trong y tế

---

## 📚 Tài liệu tham khảo

### 🔗 Links hữu ích
- **GitHub Repository**: [ImgAE-Dx](https://github.com/kinhluan/ImgAE-Dx)
- **Google Colab**: [colab.research.google.com](https://colab.research.google.com/)
- **Dataset**: [NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### 📖 Documentation khác
- [Training Guide](TRAINING_GUIDE.md) - Hướng dẫn huấn luyện chi tiết
- [Architecture Overview](../architecture/ARCHITECTURE.md) - Kiến trúc dự án
- [Project Journey](../research/PROJECT_JOURNEY.md) - Lộ trình nghiên cứu

---

## 🆘 Hỗ trợ

### 💬 Báo lỗi hoặc đặt câu hỏi
- **GitHub Issues**: [Tạo issue mới](https://github.com/kinhluan/ImgAE-Dx/issues)
- **Stack Overflow**: Tag `google-colaboratory` + `pytorch`

### 📧 Liên hệ
Nếu gặp khó khăn, hãy tạo issue trên GitHub với thông tin:
- Screenshot lỗi
- Cấu hình đã sử dụng  
- Bước đang thực hiện

---

**🎉 Chúc bạn nghiên cứu thành công với ImgAE-Dx trên Google Colab! 🧠🔬**