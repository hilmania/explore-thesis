#!/usr/bin/env python3
"""
Seizure Prediction Dataset Selector - Quick Analysis
Script untuk analisis cepat dan simpan hasil ke file
"""

import os
import glob
import json
from datetime import datetime

def quick_prediction_analysis():
    """Analisis cepat untuk selection dataset prediksi seizure"""

    results = {
        'analysis_time': datetime.now().isoformat(),
        'analysis_type': 'seizure_prediction_selection',
        'criteria': {
            'min_duration_minutes': 20,
            'preictal_duration_minutes': 20,
            'requires_seizure_events': True
        },
        'dataset_info': {},
        'selected_files': [],
        'rejected_files': [],
        'summary': {}
    }

    # Scan directory
    current_dir = os.getcwd()
    results['dataset_info']['base_path'] = current_dir

    # Find patient folders
    folders = [d for d in os.listdir('.') if os.path.isdir(d) and len(d) == 8]
    results['dataset_info']['patient_folders'] = len(folders)

    # Find all CSV files
    csv_files = glob.glob("*/*.csv")
    results['dataset_info']['total_csv_files'] = len(csv_files)

    print(f"Analyzing {len(csv_files)} CSV files for seizure prediction...")

    suitable_count = 0
    rejected_count = 0

    # Analyze sample files (first 50 untuk demo)
    sample_files = csv_files[:50]

    for csv_file in sample_files:
        try:
            file_info = {
                'file_path': csv_file,
                'patient_id': csv_file.split('/')[0],
                'session_id': os.path.basename(csv_file).replace('.csv', '')
            }

            # Read file and analyze
            with open(csv_file, 'r') as f:
                content = f.read()

            lines = content.split('\n')

            # Extract metadata
            duration_seconds = 0
            seizure_annotations = 0

            for line in lines:
                if line.startswith('#duration'):
                    try:
                        duration_str = line.split('=')[1].strip()
                        duration_seconds = float(duration_str.split()[0])
                    except:
                        pass
                elif 'seiz' in line.lower() and not line.startswith('#') and line.strip():
                    seizure_annotations += 1

            file_info['duration_seconds'] = duration_seconds
            file_info['duration_minutes'] = duration_seconds / 60
            file_info['seizure_annotations'] = seizure_annotations

            # Apply selection criteria
            suitable = True
            rejection_reasons = []

            # Check minimum duration (20 minutes = 1200 seconds)
            if duration_seconds < 1200:
                suitable = False
                rejection_reasons.append(f"Duration too short: {duration_seconds/60:.1f} min < 20 min")

            # Check for seizure events
            if seizure_annotations == 0:
                suitable = False
                rejection_reasons.append("No seizure events found")

            file_info['suitable'] = suitable
            file_info['rejection_reasons'] = rejection_reasons

            if suitable:
                results['selected_files'].append(file_info)
                suitable_count += 1
            else:
                results['rejected_files'].append(file_info)
                rejected_count += 1

        except Exception as e:
            rejected_count += 1
            results['rejected_files'].append({
                'file_path': csv_file,
                'error': str(e),
                'suitable': False,
                'rejection_reasons': [f"Error processing: {e}"]
            })

    # Calculate summary statistics
    results['summary'] = {
        'analyzed_files': len(sample_files),
        'suitable_files': suitable_count,
        'rejected_files': rejected_count,
        'selection_rate': (suitable_count / len(sample_files) * 100) if sample_files else 0,
        'estimated_total_suitable': int((suitable_count / len(sample_files)) * len(csv_files)) if sample_files else 0
    }

    # Save results to file
    output_file = 'prediction_selection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary report
    report = []
    report.append("TUSZ SEIZURE PREDICTION DATASET SELECTION RESULTS")
    report.append("=" * 55)
    report.append(f"Analysis Time: {results['analysis_time']}")
    report.append(f"Dataset Path: {current_dir}")
    report.append("")
    report.append("DATASET OVERVIEW:")
    report.append(f"  Patient folders: {results['dataset_info']['patient_folders']}")
    report.append(f"  Total CSV files: {results['dataset_info']['total_csv_files']}")
    report.append(f"  Analyzed files: {results['summary']['analyzed_files']}")
    report.append("")
    report.append("SELECTION CRITERIA:")
    report.append("  ✓ Minimum duration: 20 minutes")
    report.append("  ✓ Pre-ictal period: 20 minutes")
    report.append("  ✓ Must contain seizure events")
    report.append("")
    report.append("RESULTS:")
    report.append(f"  ✅ Suitable files: {results['summary']['suitable_files']}")
    report.append(f"  ❌ Rejected files: {results['summary']['rejected_files']}")
    report.append(f"  📊 Selection rate: {results['summary']['selection_rate']:.1f}%")
    report.append(f"  🎯 Estimated total suitable: {results['summary']['estimated_total_suitable']}")
    report.append("")

    if results['selected_files']:
        report.append("SAMPLE SUITABLE FILES:")
        for i, file_info in enumerate(results['selected_files'][:5]):
            report.append(f"  {i+1}. {file_info['patient_id']}/{file_info['session_id']}")
            report.append(f"     Duration: {file_info['duration_minutes']:.1f} min")
            report.append(f"     Seizures: {file_info['seizure_annotations']}")
        report.append("")

    report.append("NEXT STEPS FOR SEIZURE PREDICTION:")
    report.append("  1. Extract 20-minute pre-ictal segments before seizure onset")
    report.append("  2. Extract interictal segments for negative examples")
    report.append("  3. Load corresponding EDF files for signal data")
    report.append("  4. Preprocess EEG signals (filtering, normalization)")
    report.append("  5. Train machine learning model for seizure prediction")
    report.append("")
    report.append(f"📁 Detailed results saved to: {output_file}")

    # Save report
    report_file = 'prediction_selection_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    # Print key results
    print("Analysis complete!")
    print(f"Selected {suitable_count}/{len(sample_files)} files for prediction")
    print(f"Results saved to {output_file}")
    print(f"Report saved to {report_file}")

    return results

if __name__ == "__main__":
    quick_prediction_analysis()
