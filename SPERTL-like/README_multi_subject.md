# EEG Multi-Subject Seizure Prediction

Implementasi model EEGNet untuk prediksi seizure menggunakan data multi-subjek dengan arsitektur CNN dan Residual blocks.

## 📁 Struktur Data Yang Dibutuhkan

```
data/
├── BM10_train.h5          # Data EEG training subjek BM10
├── BM10_train.csv         # Label training subjek BM10
├── BM10_validation.h5     # Data EEG validation subjek BM10
├── BM10_validation.csv    # Label validation subjek BM10
├── BM10_testing.h5        # Data EEG testing subjek BM10
├── BM10_testing.csv       # Label testing subjek BM10
├── BM11_train.h5          # Data EEG training subjek BM11
├── BM11_train.csv         # Label training subjek BM11
├── BM11_validation.h5     # Data EEG validation subjek BM11
├── BM11_validation.csv    # Label validation subjek BM11
├── BM11_testing.h5        # Data EEG testing subjek BM11
├── BM11_testing.csv       # Label testing subjek BM11
├── BM12_train.h5          # Data EEG training subjek BM12
├── BM12_train.csv         # Label training subjek BM12
├── BM12_validation.h5     # Data EEG validation subjek BM12
├── BM12_validation.csv    # Label validation subjek BM12
├── BM12_testing.h5        # Data EEG testing subjek BM12
└── BM12_testing.csv       # Label testing subjek BM12
```

### Format File:

#### HDF5 Files (.h5):
- **Dataset name**: `'eeg'`
- **Shape**: `[num_samples, time_points, channels]` atau `[num_samples, channels, time_points]`
- **Data type**: `float32` (recommended)
- **Example**: Shape `[1000, 1280, 20]` = 1000 samples, 1280 time points, 20 channels

#### CSV Files (.csv):
- **Column**: `'label'` atau `'labels'` (atau kolom pertama akan digunakan)
- **Values**: `0` (no seizure) dan `1` (seizure)
- **Length**: Harus sama dengan jumlah samples di file HDF5 yang sesuai

## 🚀 Setup dan Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validasi Data

Sebelum training, validasi struktur data Anda:

```bash
python validate_data.py
```

Script ini akan:
- ✅ Mengecek keberadaan semua file yang dibutuhkan
- 📊 Menampilkan statistik data (shape, jumlah samples, distribusi label)
- ❌ Melaporkan file yang hilang atau bermasalah

## 📋 File Utama

### Core Files:
- `eeg_model.py` - Model EEGNet dengan arsitektur CNN + Residual blocks
- `eeg_dataset_multi.py` - Dataset class untuk handle multiple subjects
- `train_eegnet_multi.py` - Training script untuk semua subjek
- `cross_validation.py` - Cross-validation dengan strategi subject-independent dan subject-dependent
- `validate_data.py` - Validasi struktur dan format data

### Original Files (untuk referensi):
- `eeg_dataset.py` - Dataset class original (single file)
- `train_eegnet.py` - Training script original (single file)
- `train_eegnet_gpu.py` - Training script dengan GPU support
- `load_npy.py` - Utility untuk load file numpy

## 🏃‍♂️ Cara Menjalankan

### 1. Training Multi-Subject (Semua Data Digabung)

```bash
python train_eegnet_multi.py
```

Features:
- Menggabungkan data dari semua subjek
- Training dengan early stopping
- Evaluasi per-subjek pada test set
- Visualisasi confusion matrix dan training history

### 2. Cross-Validation Comprehensive

```bash
python cross_validation.py
```

Features:
- **Subject-Independent CV**: Leave-One-Subject-Out (LOSO)
- **Subject-Dependent Evaluation**: Train/val/test pada subjek yang sama
- Statistik lengkap dan visualisasi
- Perbandingan performa antar strategi

### 3. Validasi Data

```bash
python validate_data.py
```

## 📊 Output yang Dihasilkan

### Training Multi-Subject:
- `best_model_multi.pth` - Model terbaik
- `evaluation_report_multi.json` - Laporan evaluasi lengkap
- `confusion_matrix_multi.png` - Confusion matrix
- `training_history_multi.png` - Grafik training loss dan validation accuracy

### Cross-Validation:
- `comprehensive_cv_results.json` - Hasil CV lengkap
- `subject_independent_cv_results.png` - Visualisasi subject-independent CV
- `subject_dependent_results.png` - Visualisasi subject-dependent evaluation

### Data Validation:
- `data_summary.json` - Ringkasan lengkap struktur data

## ⚙️ Konfigurasi

### Training Parameters (train_eegnet_multi.py):
```python
data_dir = 'data'                    # Directory data
subjects = ['BM10', 'BM11', 'BM12']  # List subjek
batch_size = 64                      # Batch size
epochs = 100                         # Maksimum epochs
learning_rate = 1e-3                 # Learning rate
patience = 10                        # Early stopping patience
```

### Cross-Validation Parameters (cross_validation.py):
```python
batch_size = 32                      # Batch size untuk CV
epochs = 30                          # Epochs untuk setiap fold
lr = 1e-3                           # Learning rate
```

## 🧠 Model Architecture (EEGNet)

```
Input: [batch_size, channels, time_points]
    ↓
Initial Conv1d (channels → 64) + BatchNorm + ReLU + Dropout
    ↓
4x Residual Blocks (64 channels each)
    ↓
Global Average Pooling
    ↓
Fully Connected (64 → num_classes)
    ↓
Output: [batch_size, 2] (No Seizure vs Seizure)
```

## 📈 Evaluation Strategies

### 1. Subject-Independent (LOSO):
- **Objektif**: Generalisasi ke subjek baru
- **Method**: Train pada 2 subjek, test pada 1 subjek
- **Hasil**: 3 folds (BM10, BM11, BM12 sebagai test subject)

### 2. Subject-Dependent:
- **Objektif**: Performa optimal per subjek
- **Method**: Train/val/test pada subjek yang sama
- **Hasil**: Evaluasi terpisah untuk setiap subjek

## 🔧 Troubleshooting

### Error: File tidak ditemukan
```bash
python validate_data.py
```
Pastikan semua 18 file (9 HDF5 + 9 CSV) ada di direktori `data/`.

### Error: Shape mismatch
Pastikan:
- Jumlah samples di HDF5 = jumlah labels di CSV
- Format data sesuai dengan yang diharapkan model

### Error: Memory insufficient
- Gunakan `EEGDatasetMultiLazy` untuk lazy loading
- Kurangi `batch_size`
- Gunakan gradient accumulation

### Error: CUDA out of memory
```python
device = torch.device('cpu')  # Gunakan CPU
```

## 📝 Customization

### Menambah Subjek Baru:
1. Tambahkan file data sesuai format: `{SUBJECT}_{split}.h5` dan `{SUBJECT}_{split}.csv`
2. Update list subjek:
```python
subjects = ['BM10', 'BM11', 'BM12', 'BM13']  # Tambah BM13
```

### Mengubah Arsitektur Model:
Edit `eeg_model.py`:
```python
class EEGNet(nn.Module):
    def __init__(self, input_channels=20, num_classes=2, p_drop=0.2):
        # Modifikasi arsitektur di sini
```

### Custom Data Preprocessing:
Edit `__getitem__` di `eeg_dataset_multi.py`:
```python
def __getitem__(self, idx):
    eeg_data = self.eeg_data[idx]

    # Tambahkan preprocessing custom di sini
    # Contoh: normalization, filtering, augmentation

    return eeg_data, label
```

## 📚 Dependencies

Lihat `requirements.txt` untuk daftar lengkap dependencies:
- `torch>=1.9.0` - Deep learning framework
- `h5py>=3.1.0` - HDF5 file handling
- `pandas>=1.3.0` - CSV dan data manipulation
- `scikit-learn>=1.0.0` - Metrics dan evaluation
- `matplotlib>=3.3.0` - Plotting
- `seaborn>=0.11.0` - Advanced visualization
- `tqdm>=4.62.0` - Progress bars
- `numpy>=1.19.0` - Array operations

## 🎯 Expected Results

### Typical Performance Ranges:
- **Subject-Independent**: Accuracy 60-80%, AUC 0.6-0.8
- **Subject-Dependent**: Accuracy 70-90%, AUC 0.7-0.9

Performance tergantung pada:
- Kualitas data dan labeling
- Karakteristik individual subjek
- Hyperparameter tuning
- Data preprocessing
