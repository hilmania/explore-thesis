# TUSZ Dataset Analyzer

Toolkit Python untuk menganalisis dataset TUSZ (Temple University Seizure Corpus), mengekstrak informasi label seizure, dan menggabungkan file EDF.

## 📋 Fitur Utama

- ✅ **Analisis Label Seizure**: Ekstrak dan analisis semua jenis label seizure dari file CSV
- ✅ **Merge EDF Files**: Gabungkan multiple file EDF menjadi satu file
- ✅ **Statistik Dataset**: Generate statistik komprehensif tentang dataset
- ✅ **Visualisasi**: Plot distribusi label seizure
- ✅ **Export Data**: Export hasil analisis ke CSV dan format ML-ready
- ✅ **Quick Analysis**: Script analisis cepat untuk exploratory data analysis
- ✅ **🎯 Seizure Prediction**: Tools khusus untuk mempersiapkan dataset prediksi seizure
- ✅ **🧠 Pre-ictal Processing**: Extract segmen pre-ictal 20 menit untuk training model

## 🎯 NEW: Seizure Prediction Tools

### Fitur Khusus untuk Prediksi Seizure:
- ⚡ **Dataset Selection**: Filter file dengan durasi ≥20 menit dan seizure events
- 🧠 **Pre-ictal Extraction**: Extract segmen 20 menit sebelum onset seizure
- 🔄 **Interictal Processing**: Extract periode tanpa seizure untuk negative examples
- ⚖️ **Balanced Dataset**: Generate dataset 1:1 ratio untuk training
- 📊 **Quality Scoring**: Score kualitas recordings berdasarkan multiple criteria
- 👥 **Patient-wise Splits**: Train/validation splits yang menghindari data leakage

## 🏷️ Jenis Label Seizure yang Didukung

| Label | Deskripsi |
|-------|-----------|
| `bckg` | Background (Non-seizure) |
| `cpsz` | Complex Partial Seizure |
| `gnsz` | Generalized Non-specific Seizure |
| `fnsz` | Focal Non-specific Seizure |
| `tnsz` | Tonic Seizure |
| `absz` | Absence Seizure |
| `mysz` | Myoclonic Seizure |
| `tcsz` | Tonic-Clonic Seizure |

## 🚀 Instalasi

### 1. Install Dependencies

```bash
# Opsi 1: Menggunakan setup script
python setup_dependencies.py

# Opsi 2: Manual install
pip install -r requirements.txt

# Opsi 3: Install satu per satu
pip install pandas numpy mne matplotlib seaborn scipy scikit-learn tqdm
```

### 2. Verifikasi Instalasi

```python
import pandas as pd
import mne
import matplotlib.pyplot as plt
print("✅ Semua library berhasil diinstall!")
```

## 📖 Cara Penggunaan

### Quick Analysis (Analisis Cepat)

Untuk analisis cepat dan exploratory:

```bash
python quick_analysis.py
```

Script ini akan:
- Scan seluruh dataset secara cepat
- Menampilkan statistik dasar
- Membuat visualisasi distribusi label
- Menganalisis sample patient

### Full Analysis (Analisis Lengkap)

```bash
# Analisis seluruh dataset
python tusz_analyzer.py --dataset_path "/path/to/tusz/dataset" --verbose

# Analisis patient tertentu
python tusz_analyzer.py --dataset_path "/path/to/dataset" --patient_id aaaaaaac --verbose

# Dengan export dan visualisasi
python tusz_analyzer.py --dataset_path "/path/to/dataset" --export_csv --export_ml --plot --output_dir ./results
```

### Merge EDF Files

```bash
# Merge semua EDF files untuk satu patient
python tusz_analyzer.py --dataset_path "/path/to/dataset" --patient_id aaaaaaac --merge_edf ./merged_output
```

## 🎯 Contoh Penggunaan dalam Python

### 1. Basic Analysis

```python
from tusz_analyzer import TUSZAnalyzer

# Initialize analyzer
analyzer = TUSZAnalyzer("/path/to/tusz/dataset")

# Scan dataset
patients = analyzer.scan_dataset()
print(f"Found {len(patients)} patients")

# Analyze specific patient
patient_data = analyzer.process_patient_data("aaaaaaac", verbose=True)

# Generate statistics
report = analyzer.generate_statistics_report()
print(report['dataset_summary'])
```

### 2. Merge EDF Files

```python
# Get EDF files for a patient
patient_data = analyzer.patient_stats["aaaaaaac"]
edf_files = patient_data['edf_files']

# Merge files
combined_raw = analyzer.merge_edf_files(edf_files)
print(f"Merged {len(edf_files)} files")
print(f"Total duration: {combined_raw.times[-1]:.1f} seconds")
```

### 3. Extract Labels for Machine Learning

```python
# Export processed labels
analyzer.export_labels_for_ml("./ml_ready_labels")

# Load processed labels
import pandas as pd
labels_df = pd.read_csv("./ml_ready_labels/patient_file_labels.csv")
print(labels_df.head())
```

## 📁 Struktur Output

```
tusz_analysis_output/
├── analysis_report.json          # Report lengkap dalam JSON
├── dataset_analysis.csv           # Data tabular untuk spreadsheet
├── label_distribution.png         # Visualisasi distribusi label
└── ml_labels/                     # Labels siap untuk ML
    ├── aaaaaaac_s001_t000_labels.csv
    ├── aaaaaaac_s001_t001_labels.csv
    └── ...
```

## 📊 Format Output Data

### CSV Export Format
```csv
file_path,patient_id,session_id,montage_type,total_duration,num_channels,bckg_count,cpsz_count,gnsz_count,...
/path/file.csv,aaaaaaac,s001_2002,02_tcp_le,301.0,22,50,20,0,...
```

### ML-Ready Labels Format
```csv
channel,start_time,stop_time,label,confidence,duration,seizure_type
FP1-F7,0.0000,36.8868,bckg,1.0000,36.8868,Background (Non-seizure)
FP1-F7,36.8868,183.3055,cpsz,1.0000,146.4187,Complex Partial Seizure
```

## 🔧 Parameter Command Line

### tusz_analyzer.py

```bash
--dataset_path    # Path ke dataset TUSZ (required)
--patient_id      # Analisis patient tertentu (optional)
--merge_edf       # Directory untuk merge EDF files
--output_dir      # Directory output (default: ./tusz_analysis_output)
--export_csv      # Export ke CSV
--export_ml       # Export labels untuk ML
--plot           # Generate visualisasi
--verbose        # Output detail
```

## 🎯 Seizure Prediction Workflow

### 1. Select Suitable Dataset

```bash
# Filter files untuk seizure prediction (durasi ≥20 min, ada seizure events)
python seizure_prediction_selector.py \
    --dataset_path "/path/to/tusz/dataset" \
    --min_duration 20 \
    --preictal_duration 20 \
    --output_dir ./prediction_dataset \
    --quality_threshold 70 \
    --verbose
```

### 2. Extract Pre-ictal Segments

```bash
# Extract 20-minute pre-ictal segments dan interictal periods
python preictal_processor.py \
    --prediction_dataset ./prediction_dataset/prediction_dataset.json \
    --output_dir ./preictal_data \
    --preictal_duration 20 \
    --interictal_duration 20 \
    --sampling_rate 250 \
    --verbose
```

### 3. Load Processed Data for Training

```python
from preictal_processor import load_processed_data
import numpy as np

# Load processed segments
data = load_processed_data('./preictal_data')
preictal_eeg = data['preictal']['data']      # Shape: (n_segments, channels, samples)
interictal_eeg = data['interictal']['data']  # Shape: (n_segments, channels, samples)

# Create training dataset
X = np.concatenate([preictal_eeg, interictal_eeg])
y = np.concatenate([
    np.ones(preictal_eeg.shape[0]),    # Pre-ictal = 1
    np.zeros(interictal_eeg.shape[0])  # Interictal = 0
])

print(f"Training data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### 4. Workflow Demo

```bash
# Lihat demo lengkap workflow prediksi seizure
python seizure_prediction_workflow.py

# Analyze hasil selection yang sudah ada
python seizure_prediction_workflow.py --analyze_results ./prediction_dataset/prediction_dataset.json
```

## 🎯 Use Cases

### 1. Exploratory Data Analysis
```bash
python quick_analysis.py
```

### 2. Dataset Statistics
```bash
python tusz_analyzer.py --dataset_path "/path/to/dataset" --export_csv --plot
```

### 3. Prepare Data for Machine Learning
```bash
python tusz_analyzer.py --dataset_path "/path/to/dataset" --export_ml --output_dir ./ml_data
```

### 4. EEG Signal Processing
```python
# Load merged EDF
import mne
raw = mne.io.read_raw_fif("merged_patient_data.fif", preload=True)

# Apply filters
raw.filter(l_freq=1.0, h_freq=40.0)

# Extract epochs around seizure events
# (combine with label information)
```

## 🛠️ Troubleshooting

### Error: "Import pandas could not be resolved"
```bash
pip install pandas numpy matplotlib seaborn mne
```

### Error: "Patient directory not found"
- Pastikan path dataset sudah benar
- Check format patient ID (8 karakter, contoh: aaaaaaac)

### Error: "No valid EDF files to merge"
- Pastikan file EDF ada di directory patient
- Check permission file

### Memory Error saat merge EDF
```python
# Load dengan preload=False untuk file besar
raw = mne.io.read_raw_edf(file_path, preload=False)
```

## 📈 Analisis Lanjutan

### 1. Time Series Analysis
```python
# Load EEG data dan labels
raw = mne.io.read_raw_edf("file.edf", preload=True)
labels_df = pd.read_csv("labels.csv")

# Extract seizure epochs
seizure_events = labels_df[labels_df['label'] != 'bckg']
```

### 2. Feature Extraction
```python
# Extract features dari EEG signals
from scipy.signal import welch

# Power spectral density
freqs, psd = welch(raw.get_data(), fs=raw.info['sfreq'])
```

### 3. Machine Learning Pipeline
```python
# Load processed labels
labels_df = pd.read_csv("ml_labels/processed_labels.csv")

# Create binary classification (seizure vs non-seizure)
labels_df['is_seizure'] = labels_df['label'] != 'bckg'
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📜 License

MIT License - see LICENSE file for details

## 📞 Support

Jika mengalami issues atau butuh bantuan:
1. Check troubleshooting section
2. Review error messages
3. Pastikan dependencies terinstall dengan benar

## 🔄 Updates

- v1.0: Initial release with basic analysis
- v1.1: Added EDF merging capability
- v1.2: Added ML-ready export format
- v1.3: Added quick analysis script
- v2.0: **NEW!** Added seizure prediction tools
  - seizure_prediction_selector.py - Dataset selection untuk prediksi
  - preictal_processor.py - Pre-ictal segment extraction
  - seizure_prediction_workflow.py - Complete workflow demo

## 📦 Complete Toolkit Files

### Core Analysis Tools:
- ✅ `tusz_analyzer.py` - Main comprehensive analyzer
- ✅ `quick_analysis.py` - Fast exploratory analysis
- ✅ `simple_analyzer.py` - No-dependency analyzer

### 🎯 Seizure Prediction Tools:
- ⚡ `seizure_prediction_selector.py` - Select files untuk prediksi (≥20 min)
- 🧠 `preictal_processor.py` - Extract pre-ictal segments (20 min before seizure)
- 🔄 `seizure_prediction_workflow.py` - Demo complete workflow

### Support Files:
- ✅ `setup_dependencies.py` - Auto installer
- ✅ `requirements.txt` - Dependencies list
- ✅ `demo_analysis.py` - Demonstration script
- ✅ `test_analysis.py` - Testing utilities
- ✅ `README.md` - This documentation

---

**Happy analyzing! 🧠✨**
