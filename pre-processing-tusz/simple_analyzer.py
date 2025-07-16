#!/usr/bin/env python3
"""
TUSZ Simple Analyzer - No External Dependencies
===============================================

Script sederhana untuk analisis dataset TUSZ tanpa menggunakan
dependencies eksternal (hanya Python standard library).

Usage:
    python simple_analyzer.py

Author: Assistant
Date: July 2025
"""

import os
import csv
from pathlib import Path
from collections import Counter, defaultdict
import json

def analyze_csv_file(csv_path):
    """
    Analyze single CSV file and extract seizure labels

    Parameters:
    -----------
    csv_path : str
        Path to CSV file

    Returns:
    --------
    dict: Analysis results
    """
    try:
        with open(csv_path, 'r') as file:
            lines = file.readlines()

        # Find metadata and header
        metadata = {}
        header_idx = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                if '=' in line:
                    key, value = line[1:].split('=', 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith('channel,'):
                header_idx = i
                break

        # Read data section
        data_lines = lines[header_idx+1:]
        labels = []
        durations = []

        for line in data_lines:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    start_time = float(parts[1])
                    stop_time = float(parts[2])
                    label = parts[3]
                    duration = stop_time - start_time

                    labels.append(label)
                    durations.append(duration)

        # Calculate statistics
        label_counts = Counter(labels)
        total_duration = float(metadata.get('duration', '0').split()[0])

        return {
            'file_path': str(csv_path),
            'metadata': metadata,
            'labels': list(set(labels)),
            'label_counts': dict(label_counts),
            'total_duration': total_duration,
            'total_annotations': len(labels),
            'success': True
        }

    except Exception as e:
        return {
            'file_path': str(csv_path),
            'error': str(e),
            'success': False
        }

def scan_patient_directory(patient_path):
    """
    Scan a single patient directory

    Parameters:
    -----------
    patient_path : Path
        Path to patient directory

    Returns:
    --------
    dict: Patient analysis results
    """
    patient_id = patient_path.name

    # Find all CSV files (excluding .csv_bi files)
    csv_files = []
    edf_files = []

    for root, dirs, files in os.walk(patient_path):
        for file in files:
            if file.endswith('.csv') and not file.endswith('.csv_bi'):
                csv_files.append(Path(root) / file)
            elif file.endswith('.edf'):
                edf_files.append(Path(root) / file)

    # Analyze each CSV file
    file_analyses = []
    all_labels = []
    total_duration = 0

    for csv_file in csv_files:
        analysis = analyze_csv_file(csv_file)
        if analysis['success']:
            file_analyses.append(analysis)
            all_labels.extend(analysis['labels'])
            total_duration += analysis['total_duration']

    # Collect session information
    sessions = set()
    montages = set()

    for csv_file in csv_files:
        parts = csv_file.parts
        if len(parts) >= 3:
            session_id = parts[-3]  # e.g., s001_2002
            montage_type = parts[-2]  # e.g., 02_tcp_le
            sessions.add(session_id)
            montages.add(montage_type)

    return {
        'patient_id': patient_id,
        'csv_files_count': len(csv_files),
        'edf_files_count': len(edf_files),
        'sessions': list(sessions),
        'montage_types': list(montages),
        'seizure_types_found': list(set(all_labels)),
        'total_duration_seconds': total_duration,
        'total_duration_minutes': total_duration / 60,
        'file_analyses': file_analyses
    }

def analyze_tusz_dataset(base_path):
    """
    Analyze complete TUSZ dataset

    Parameters:
    -----------
    base_path : str
        Path to TUSZ dataset base directory

    Returns:
    --------
    dict: Complete dataset analysis
    """
    base_path = Path(base_path)

    print("🧠 TUSZ Simple Analyzer")
    print("=" * 50)
    print(f"📂 Dataset path: {base_path}")

    # Find all patient directories
    patient_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and len(item.name) == 8:  # Patient ID format
            patient_dirs.append(item)

    print(f"👥 Found {len(patient_dirs)} patient directories")

    # Analyze each patient
    patient_analyses = []
    all_seizure_types = set()
    total_files = 0
    total_duration = 0
    global_label_counts = Counter()

    for i, patient_dir in enumerate(patient_dirs):
        print(f"🔄 Processing patient {i+1}/{len(patient_dirs)}: {patient_dir.name}")

        patient_analysis = scan_patient_directory(patient_dir)
        patient_analyses.append(patient_analysis)

        # Update global statistics
        all_seizure_types.update(patient_analysis['seizure_types_found'])
        total_files += patient_analysis['csv_files_count']
        total_duration += patient_analysis['total_duration_seconds']

        # Count labels across all files
        for file_analysis in patient_analysis['file_analyses']:
            for label, count in file_analysis['label_counts'].items():
                global_label_counts[label] += count

    # Seizure type descriptions
    seizure_descriptions = {
        'bckg': 'Background (Non-seizure)',
        'cpsz': 'Complex Partial Seizure',
        'gnsz': 'Generalized Non-specific Seizure',
        'fnsz': 'Focal Non-specific Seizure',
        'tnsz': 'Tonic Seizure',
        'absz': 'Absence Seizure',
        'mysz': 'Myoclonic Seizure',
        'tcsz': 'Tonic-Clonic Seizure'
    }

    # Create summary
    summary = {
        'dataset_info': {
            'base_path': str(base_path),
            'total_patients': len(patient_dirs),
            'total_csv_files': total_files,
            'total_duration_hours': total_duration / 3600,
            'seizure_types_found': list(all_seizure_types)
        },
        'label_distribution': dict(global_label_counts),
        'seizure_descriptions': seizure_descriptions,
        'patient_analyses': patient_analyses
    }

    return summary

def print_analysis_results(analysis):
    """Print analysis results in a readable format"""

    print("\n📊 DATASET SUMMARY")
    print("=" * 50)

    info = analysis['dataset_info']
    print(f"👥 Total Patients: {info['total_patients']}")
    print(f"📁 Total CSV Files: {info['total_csv_files']}")
    print(f"⏱️  Total Duration: {info['total_duration_hours']:.1f} hours")
    print(f"🧠 Seizure Types Found: {len(info['seizure_types_found'])}")

    print("\n🏷️  SEIZURE LABEL DISTRIBUTION")
    print("=" * 50)

    label_dist = analysis['label_distribution']
    total_labels = sum(label_dist.values())

    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
        description = analysis['seizure_descriptions'].get(label, 'Unknown')
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"{label:4s} ({description:30s}): {count:,} ({percentage:.1f}%)")

    print(f"\nTotal annotations: {total_labels:,}")

    # Top patients by duration
    print("\n👥 TOP 10 PATIENTS BY DURATION")
    print("=" * 50)

    patients_by_duration = sorted(
        analysis['patient_analyses'],
        key=lambda x: x['total_duration_minutes'],
        reverse=True
    )[:10]

    for i, patient in enumerate(patients_by_duration, 1):
        print(f"{i:2d}. {patient['patient_id']}: {patient['total_duration_minutes']:.1f} min "
              f"({patient['csv_files_count']} files, {len(patient['sessions'])} sessions)")

def save_analysis_results(analysis, output_file):
    """Save analysis results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n💾 Analysis results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def export_to_csv(analysis, output_file):
    """Export analysis results to CSV format"""
    try:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'patient_id', 'csv_files', 'edf_files', 'sessions', 'montage_types',
                'duration_minutes', 'seizure_types'
            ]

            # Add label count columns
            all_labels = set()
            for patient in analysis['patient_analyses']:
                for file_analysis in patient['file_analyses']:
                    all_labels.update(file_analysis['label_counts'].keys())

            for label in sorted(all_labels):
                fieldnames.append(f'{label}_count')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for patient in analysis['patient_analyses']:
                row = {
                    'patient_id': patient['patient_id'],
                    'csv_files': patient['csv_files_count'],
                    'edf_files': patient['edf_files_count'],
                    'sessions': len(patient['sessions']),
                    'montage_types': '; '.join(patient['montage_types']),
                    'duration_minutes': patient['total_duration_minutes'],
                    'seizure_types': '; '.join(patient['seizure_types_found'])
                }

                # Add label counts
                patient_label_counts = Counter()
                for file_analysis in patient['file_analyses']:
                    for label, count in file_analysis['label_counts'].items():
                        patient_label_counts[label] += count

                for label in sorted(all_labels):
                    row[f'{label}_count'] = patient_label_counts.get(label, 0)

                writer.writerow(row)

        print(f"📄 CSV export saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")

def main():
    """Main function"""
    # Ganti path ini sesuai dengan lokasi dataset Anda
    DATASET_PATH = "/Volumes/Hilmania/TUH SZ/v2.0.3/edf/train"

    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path tidak ditemukan: {DATASET_PATH}")
        print("Silakan edit variable DATASET_PATH di script ini")
        return

    try:
        # Analyze dataset
        analysis_results = analyze_tusz_dataset(DATASET_PATH)

        # Print results
        print_analysis_results(analysis_results)

        # Save results
        save_analysis_results(analysis_results, 'tusz_analysis_results.json')
        export_to_csv(analysis_results, 'tusz_analysis_summary.csv')

        print("\n✅ Analysis completed successfully!")
        print("\n💡 FILES CREATED:")
        print("  📄 tusz_analysis_results.json - Full analysis results")
        print("  📊 tusz_analysis_summary.csv - Summary table")

        print("\n🚀 NEXT STEPS:")
        print("  1. Review the analysis results")
        print("  2. Install MNE-Python for EDF processing: pip install mne")
        print("  3. Use tusz_analyzer.py for advanced features")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
