#!/usr/bin/env python3
"""
TUSZ Quick Analyzer - Contoh Penggunaan Sederhana
================================================

Script sederhana untuk analisis cepat dataset TUSZ.
Cocok untuk exploratory data analysis.

Usage:
    python quick_analysis.py

Author: Assistant
Date: July 2025
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def quick_scan_tusz(base_path):
    """
    Scan cepat dataset TUSZ untuk mendapatkan overview

    Parameters:
    -----------
    base_path : str
        Path ke direktori dataset TUSZ
    """
    print("🧠 TUSZ Quick Scanner")
    print("=" * 40)

    base_path = Path(base_path)

    # Statistik dasar
    patients = []
    all_labels = []
    total_files = 0
    total_duration = 0

    print("🔍 Scanning directories...")

    # Scan semua directory patient
    for patient_dir in base_path.iterdir():
        if patient_dir.is_dir() and len(patient_dir.name) == 8:
            patients.append(patient_dir.name)

            # Scan CSV files dalam patient directory
            csv_files = list(patient_dir.rglob('*.csv'))
            csv_files = [f for f in csv_files if not f.name.endswith('.csv_bi')]

            for csv_file in csv_files:
                try:
                    # Baca file CSV
                    with open(csv_file, 'r') as f:
                        lines = f.readlines()

                    # Cari header
                    header_idx = 0
                    duration = 0
                    for i, line in enumerate(lines):
                        if line.startswith('# duration'):
                            duration_str = line.split('=')[1].strip()
                            duration = float(duration_str.split()[0])
                        elif line.startswith('channel,'):
                            header_idx = i
                            break

                    # Baca data
                    df = pd.read_csv(csv_file, skiprows=header_idx)
                    labels = df['label'].tolist()
                    all_labels.extend(labels)

                    total_files += 1
                    total_duration += duration

                except Exception as e:
                    print(f"⚠️ Skipping {csv_file}: {e}")
                    continue

    # Analisis hasil
    print(f"\n📊 HASIL SCAN")
    print("-" * 30)
    print(f"👥 Total Patients: {len(patients)}")
    print(f"📁 Total CSV Files: {total_files}")
    print(f"⏱️ Total Duration: {total_duration/3600:.1f} jam")

    # Distribusi label
    label_counts = Counter(all_labels)
    print(f"\n🏷️ DISTRIBUSI LABEL")
    print("-" * 30)

    seizure_types = {
        'bckg': 'Background (Non-seizure)',
        'cpsz': 'Complex Partial Seizure',
        'gnsz': 'Generalized Non-specific Seizure',
        'fnsz': 'Focal Non-specific Seizure',
        'tnsz': 'Tonic Seizure',
        'absz': 'Absence Seizure',
        'mysz': 'Myoclonic Seizure',
        'tcsz': 'Tonic-Clonic Seizure'
    }

    for label, count in label_counts.most_common():
        description = seizure_types.get(label, 'Unknown')
        percentage = (count / len(all_labels)) * 100
        print(f"{label:4s} ({description:25s}): {count:,} ({percentage:.1f}%)")

    # Buat plot sederhana
    if len(label_counts) > 0:
        print("\n📈 Membuat visualisasi...")
        create_simple_plot(label_counts, seizure_types)

    return {
        'patients': patients,
        'total_files': total_files,
        'total_duration': total_duration,
        'label_distribution': label_counts
    }

def create_simple_plot(label_counts, seizure_types):
    """Buat plot distribusi label"""
    try:
        plt.figure(figsize=(12, 6))

        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(labels, counts, color=colors)
        plt.title('Distribusi Label Seizure')
        plt.xlabel('Label')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=45)

        # Tambah nilai di atas bar
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=8)

        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Persentase Label Seizure')

        plt.tight_layout()
        plt.savefig('tusz_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Plot disimpan sebagai 'tusz_label_distribution.png'")

    except Exception as e:
        print(f"⚠️ Tidak bisa membuat plot: {e}")

def analyze_single_patient(base_path, patient_id):
    """
    Analisis detail untuk satu patient

    Parameters:
    -----------
    base_path : str
        Path ke direktori dataset
    patient_id : str
        ID patient (8 karakter)
    """
    print(f"\n👤 ANALISIS PATIENT: {patient_id}")
    print("=" * 40)

    patient_path = Path(base_path) / patient_id
    if not patient_path.exists():
        print(f"❌ Patient {patient_id} tidak ditemukan!")
        return

    # Cari semua file CSV dan EDF
    csv_files = list(patient_path.rglob('*.csv'))
    csv_files = [f for f in csv_files if not f.name.endswith('.csv_bi')]
    edf_files = list(patient_path.rglob('*.edf'))

    print(f"📁 CSV files: {len(csv_files)}")
    print(f"📊 EDF files: {len(edf_files)}")

    # Analisis per session
    sessions = {}
    for csv_file in csv_files:
        parts = csv_file.parts
        session_id = parts[-3]  # s001_2002 format
        montage = parts[-2]     # montage type

        if session_id not in sessions:
            sessions[session_id] = {}
        if montage not in sessions[session_id]:
            sessions[session_id][montage] = []

        sessions[session_id][montage].append(csv_file)

    print(f"\n📅 SESSIONS:")
    for session_id, montages in sessions.items():
        print(f"  {session_id}:")
        for montage, files in montages.items():
            print(f"    {montage}: {len(files)} files")

    # Analisis labels untuk patient ini
    all_labels = []
    total_duration = 0

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Extract duration
            duration = 0
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('# duration'):
                    duration_str = line.split('=')[1].strip()
                    duration = float(duration_str.split()[0])
                elif line.startswith('channel,'):
                    header_idx = i
                    break

            df = pd.read_csv(csv_file, skiprows=header_idx)
            all_labels.extend(df['label'].tolist())
            total_duration += duration

        except Exception as e:
            print(f"⚠️ Error reading {csv_file}: {e}")

    print(f"\n📊 STATISTIK PATIENT:")
    print(f"⏱️ Total duration: {total_duration/60:.1f} menit")

    label_counts = Counter(all_labels)
    for label, count in label_counts.most_common():
        percentage = (count / len(all_labels)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

def main():
    """Main function untuk quick analysis"""
    # Ganti path ini sesuai dengan lokasi dataset Anda
    DATASET_PATH = "/Volumes/Hilmania/TUH SZ/v2.0.3/edf/train"

    # Quick scan seluruh dataset
    results = quick_scan_tusz(DATASET_PATH)

    # Contoh analisis patient tertentu
    # Ganti 'aaaaaaac' dengan patient ID yang ingin dianalisis
    if results['patients']:
        sample_patient = results['patients'][0]  # Ambil patient pertama
        analyze_single_patient(DATASET_PATH, sample_patient)

    print("\n✅ Quick analysis selesai!")
    print("\n💡 TIPS:")
    print("- Untuk analisis lengkap, gunakan: python tusz_analyzer.py --dataset_path [PATH]")
    print("- Untuk merge EDF files: python tusz_analyzer.py --dataset_path [PATH] --patient_id [ID] --merge_edf [OUTPUT_DIR]")
    print("- Untuk export ke CSV: python tusz_analyzer.py --dataset_path [PATH] --export_csv")

if __name__ == "__main__":
    main()
