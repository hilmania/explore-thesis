#!/usr/bin/env python3
"""
TUSZ Seizure Prediction Dataset Selector - Updated Version
Menganalisis dataset TUSZ dengan struktur hierarki yang benar untuk memilih
file yang sesuai untuk prediksi seizure dengan fase pre-ictal 20 menit.
"""

import os
import glob
import json
import csv
from pathlib import Path
from datetime import datetime

def analyze_seizure_prediction_dataset():
    """Analisis dataset untuk seleksi file prediksi seizure"""

    results = {
        'analysis_time': datetime.now().isoformat(),
        'analysis_type': 'seizure_prediction_selection',
        'criteria': {
            'min_duration_minutes': 20,
            'min_duration_seconds': 1200,
            'preictal_duration_minutes': 20,
            'preictal_duration_seconds': 1200,
            'requires_seizure_events': True
        },
        'dataset_structure': {},
        'selected_files': [],
        'rejected_files': [],
        'patient_analysis': {},
        'summary': {}
    }

    print("🎯 TUSZ Seizure Prediction Dataset Analysis")
    print("=" * 50)
    print("Criteria: Duration ≥20 min, Pre-ictal ≥20 min, Has seizures")
    print()

    # Scan dataset structure
    base_path = Path(".")
    patient_folders = [d for d in base_path.iterdir() if d.is_dir() and len(d.name) == 8]

    print(f"📁 Found {len(patient_folders)} patient folders")

    # Find all CSV files with correct structure
    csv_files = []
    for pattern in ["*/*/*/*.csv", "*/*/*/*/*.csv"]:
        csv_files.extend(glob.glob(pattern))

    # Filter out backup files (.csv_bi)
    csv_files = [f for f in csv_files if not f.endswith('.csv_bi')]

    print(f"📄 Found {len(csv_files)} CSV annotation files")
    print()

    suitable_files = []
    rejected_files = []
    patient_stats = {}

    # Process sample of files (first 100 for demo)
    sample_files = csv_files[:100]
    print(f"🔍 Analyzing {len(sample_files)} sample files...")

    for i, csv_file in enumerate(sample_files):
        if i % 20 == 0:
            print(f"   Processing {i+1}/{len(sample_files)}...")

        try:
            # Parse file path to extract patient and session info
            path_parts = csv_file.split('/')
            patient_id = path_parts[0]
            session_id = path_parts[1] if len(path_parts) > 1 else "unknown"
            file_name = os.path.basename(csv_file)

            # Initialize patient stats
            if patient_id not in patient_stats:
                patient_stats[patient_id] = {
                    'total_files': 0,
                    'suitable_files': 0,
                    'total_duration': 0,
                    'total_seizures': 0
                }

            patient_stats[patient_id]['total_files'] += 1

            # Read CSV file and extract metadata
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Parse metadata from header
            duration_seconds = 0
            for line in lines:
                if line.startswith('# duration'):
                    try:
                        duration_str = line.split('=')[1].strip()
                        duration_seconds = float(duration_str.split()[0])
                        break
                    except:
                        pass

            # Read annotation data
            header_line = None
            for i, line in enumerate(lines):
                if line.startswith('channel,'):
                    header_line = i
                    break

            if header_line is None:
                continue

            # Read annotation data using csv module
            annotation_data = []
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                reader = csv.DictReader(lines[header_line:])
                for row in reader:
                    annotation_data.append(row)

            # Analyze seizure events
            seizure_events = []
            seizure_types = set()

            for row in annotation_data:
                if row['label'] != 'bckg':
                    seizure_types.add(row['label'])
                    # Simple seizure event detection
                    start_time = float(row['start_time'])
                    end_time = float(row['stop_time'])

                    # Group similar seizure events
                    found_existing = False
                    for event in seizure_events:
                        if (event['type'] == row['label'] and
                            abs(event['start_time'] - start_time) < 5.0):
                            # Extend existing event
                            event['end_time'] = max(event['end_time'], end_time)
                            found_existing = True
                            break

                    if not found_existing:
                        seizure_events.append({
                            'type': row['label'],
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time
                        })

            seizure_types = list(seizure_types)

            # Create file analysis result
            file_info = {
                'file_path': csv_file,
                'patient_id': patient_id,
                'session_id': session_id,
                'file_name': file_name,
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_seconds / 60,
                'seizure_events': len(seizure_events),
                'seizure_types': list(seizure_types),
                'seizure_details': seizure_events
            }

            # Update patient statistics
            patient_stats[patient_id]['total_duration'] += duration_seconds
            patient_stats[patient_id]['total_seizures'] += len(seizure_events)

            # Apply selection criteria
            suitable = True
            rejection_reasons = []

            # Criterion 1: Minimum duration (20 minutes = 1200 seconds)
            if duration_seconds < 1200:
                suitable = False
                rejection_reasons.append(f"Duration too short: {duration_seconds/60:.1f} min < 20 min")

            # Criterion 2: Must have seizure events
            if len(seizure_events) == 0:
                suitable = False
                rejection_reasons.append("No seizure events found")

            # Criterion 3: First seizure must be ≥20 min from start (for pre-ictal)
            if seizure_events and suitable:
                first_seizure_time = min([event['start_time'] for event in seizure_events])
                if first_seizure_time < 1200:  # 20 minutes
                    suitable = False
                    rejection_reasons.append(f"First seizure at {first_seizure_time/60:.1f} min, need ≥20 min pre-ictal")

            file_info['suitable'] = suitable
            file_info['rejection_reasons'] = rejection_reasons

            if suitable:
                suitable_files.append(file_info)
                patient_stats[patient_id]['suitable_files'] += 1
            else:
                rejected_files.append(file_info)

        except Exception as e:
            rejected_files.append({
                'file_path': csv_file,
                'error': str(e),
                'suitable': False,
                'rejection_reasons': [f"Error processing: {e}"]
            })

    # Compile results
    results['selected_files'] = suitable_files
    results['rejected_files'] = rejected_files
    results['patient_analysis'] = patient_stats

    # Calculate summary statistics
    total_analyzed = len(sample_files)
    total_suitable = len(suitable_files)
    total_rejected = len(rejected_files)

    results['summary'] = {
        'total_patients': len(patient_stats),
        'total_analyzed_files': total_analyzed,
        'total_suitable_files': total_suitable,
        'total_rejected_files': total_rejected,
        'selection_rate_percent': (total_suitable / total_analyzed * 100) if total_analyzed > 0 else 0,
        'estimated_suitable_in_full_dataset': int((total_suitable / total_analyzed) * len(csv_files)) if total_analyzed > 0 else 0
    }

    # Print results
    print("\n📊 ANALYSIS RESULTS:")
    print(f"   Total patients analyzed: {len(patient_stats)}")
    print(f"   Total files analyzed: {total_analyzed}")
    print(f"   ✅ Suitable files: {total_suitable}")
    print(f"   ❌ Rejected files: {total_rejected}")
    print(f"   📈 Selection rate: {results['summary']['selection_rate_percent']:.1f}%")
    print(f"   🎯 Estimated suitable files in full dataset: {results['summary']['estimated_suitable_in_full_dataset']}")

    # Show top rejection reasons
    rejection_counts = {}
    for file_info in rejected_files:
        for reason in file_info.get('rejection_reasons', []):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    if rejection_counts:
        print(f"\n❌ TOP REJECTION REASONS:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {reason}: {count} files")

    # Show sample suitable files
    if suitable_files:
        print(f"\n✅ SAMPLE SUITABLE FILES FOR PREDICTION:")
        for i, file_info in enumerate(suitable_files[:5]):
            print(f"   {i+1}. {file_info['patient_id']}/{file_info['session_id']}")
            print(f"      Duration: {file_info['duration_minutes']:.1f} min")
            print(f"      Seizures: {file_info['seizure_events']} ({', '.join(file_info['seizure_types'])})")
            if file_info['seizure_details']:
                first_seizure = min([s['start_time'] for s in file_info['seizure_details']])
                print(f"      Pre-ictal available: {first_seizure/60:.1f} min")
            print()

    # Patient-level analysis
    print(f"👥 PATIENT-LEVEL ANALYSIS:")
    suitable_patients = [p for p, stats in patient_stats.items() if stats['suitable_files'] > 0]
    print(f"   Patients with suitable data: {len(suitable_patients)}/{len(patient_stats)}")

    if suitable_patients:
        print(f"   Top patients by suitable files:")
        sorted_patients = sorted([(p, stats['suitable_files']) for p, stats in patient_stats.items()],
                               key=lambda x: x[1], reverse=True)
        for i, (patient, count) in enumerate(sorted_patients[:5]):
            if count > 0:
                total_duration = patient_stats[patient]['total_duration'] / 60
                print(f"     {i+1}. {patient}: {count} files, {total_duration:.1f} min total")

    # Save results
    output_file = 'seizure_prediction_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary CSV for suitable files
    if suitable_files:
        summary_data = []
        for file_info in suitable_files:
            summary_data.append({
                'patient_id': file_info['patient_id'],
                'session_id': file_info['session_id'],
                'file_path': file_info['file_path'],
                'duration_minutes': file_info['duration_minutes'],
                'seizure_count': file_info['seizure_events'],
                'seizure_types': ', '.join(file_info['seizure_types']),
                'preictal_minutes': min([s['start_time'] for s in file_info['seizure_details']])/60 if file_info['seizure_details'] else 0
            })

        # Create summary CSV for suitable files manually
        csv_filename = 'suitable_files_for_prediction.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['patient_id', 'session_id', 'file_path', 'duration_minutes',
                         'seizure_count', 'seizure_types', 'preictal_minutes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for file_info in suitable_files:
                writer.writerow({
                    'patient_id': file_info['patient_id'],
                    'session_id': file_info['session_id'],
                    'file_path': file_info['file_path'],
                    'duration_minutes': file_info['duration_minutes'],
                    'seizure_count': file_info['seizure_events'],
                    'seizure_types': ', '.join(file_info['seizure_types']),
                    'preictal_minutes': min([s['start_time'] for s in file_info['seizure_details']])/60 if file_info['seizure_details'] else 0
                })
        print(f"\n💾 Results saved:")
        print(f"   📊 {output_file} - Complete analysis results")
        print(f"   📋 suitable_files_for_prediction.csv - Summary of suitable files")

    print(f"\n🚀 NEXT STEPS FOR SEIZURE PREDICTION:")
    print(f"   1. Load EDF files corresponding to suitable CSV files")
    print(f"   2. Extract 20-minute pre-ictal segments before first seizure")
    print(f"   3. Extract interictal segments (normal periods) for negative examples")
    print(f"   4. Preprocess EEG signals (bandpass filter, artifact removal)")
    print(f"   5. Feature extraction (spectral, temporal, connectivity features)")
    print(f"   6. Train machine learning model (SVM, Random Forest, Deep Learning)")
    print(f"   7. Evaluate model performance (sensitivity, specificity, false positive rate)")

    return results

if __name__ == "__main__":
    analyze_seizure_prediction_dataset()
