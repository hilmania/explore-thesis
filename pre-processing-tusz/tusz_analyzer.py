#!/usr/bin/env python3
"""
TUSZ Dataset Analyzer and EDF Processor
=======================================

Script untuk menganalisis dataset TUSZ (Temple University Seizure Corpus):
1. Mengekstrak informasi label seizure dari file CSV
2. Menggabungkan multiple file EDF
3. Analisis distribusi label dan statistik dataset
4. Export data untuk machine learning

Author: Assistant
Date: July 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import mne
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class TUSZAnalyzer:
    """Class untuk menganalisis dataset TUSZ"""

    def __init__(self, base_path):
        """
        Initialize TUSZ Analyzer

        Parameters:
        -----------
        base_path : str
            Path ke direktori dataset TUSZ
        """
        self.base_path = Path(base_path)
        self.seizure_types = {
            'bckg': 'Background (Non-seizure)',
            'cpsz': 'Complex Partial Seizure',
            'gnsz': 'Generalized Non-specific Seizure',
            'fnsz': 'Focal Non-specific Seizure',
            'tnsz': 'Tonic Seizure',
            'absz': 'Absence Seizure',
            'mysz': 'Myoclonic Seizure',
            'tcsz': 'Tonic-Clonic Seizure'
        }

        # Statistics storage
        self.label_stats = defaultdict(int)
        self.patient_stats = defaultdict(dict)
        self.file_info = []

    def scan_dataset(self):
        """Scan seluruh dataset untuk mendapatkan informasi struktur"""
        print("🔍 Scanning dataset structure...")

        patients = []
        for patient_dir in self.base_path.iterdir():
            if patient_dir.is_dir() and len(patient_dir.name) == 8:  # Patient ID format
                patients.append(patient_dir.name)

        print(f"📊 Found {len(patients)} patients in dataset")
        return sorted(patients)

    def extract_csv_labels(self, csv_file_path):
        """
        Extract seizure labels from CSV file

        Parameters:
        -----------
        csv_file_path : Path
            Path ke file CSV

        Returns:
        --------
        dict: Informasi label dan timing
        """
        try:
            # Read CSV file, skip header comments
            with open(csv_file_path, 'r') as f:
                lines = f.readlines()

            # Find header line
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('channel,'):
                    header_idx = i
                    break

            # Read data from header
            df = pd.read_csv(csv_file_path, skiprows=header_idx)

            # Extract metadata from header comments
            metadata = {}
            for line in lines[:header_idx]:
                if line.startswith('#'):
                    line = line.strip('# \n')
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()

            # Analyze labels
            labels = df['label'].unique()
            label_counts = df['label'].value_counts().to_dict()

            # Calculate duration for each label
            label_durations = {}
            for label in labels:
                label_df = df[df['label'] == label]
                total_duration = (label_df['stop_time'] - label_df['start_time']).sum()
                label_durations[label] = total_duration

            return {
                'file_path': str(csv_file_path),
                'metadata': metadata,
                'labels': list(labels),
                'label_counts': label_counts,
                'label_durations': label_durations,
                'total_duration': float(metadata.get('duration', '0').split()[0]),
                'num_channels': len(df['channel'].unique())
            }

        except Exception as e:
            print(f"❌ Error processing {csv_file_path}: {e}")
            return None

    def process_patient_data(self, patient_id, verbose=False):
        """
        Process all data for a specific patient

        Parameters:
        -----------
        patient_id : str
            Patient identifier
        verbose : bool
            Print detailed information

        Returns:
        --------
        dict: Patient data summary
        """
        patient_path = self.base_path / patient_id
        if not patient_path.exists():
            print(f"❌ Patient directory not found: {patient_id}")
            return None

        patient_data = {
            'patient_id': patient_id,
            'sessions': [],
            'total_files': 0,
            'total_duration': 0,
            'seizure_types_found': set(),
            'edf_files': [],
            'csv_files': []
        }

        if verbose:
            print(f"\n👤 Processing patient: {patient_id}")

        # Iterate through sessions
        for session_dir in patient_path.iterdir():
            if session_dir.is_dir():
                session_data = {
                    'session_id': session_dir.name,
                    'montages': []
                }

                # Iterate through montages
                for montage_dir in session_dir.iterdir():
                    if montage_dir.is_dir():
                        montage_data = {
                            'montage_type': montage_dir.name,
                            'files': []
                        }

                        # Process CSV and EDF files
                        csv_files = list(montage_dir.glob('*.csv'))
                        edf_files = list(montage_dir.glob('*.edf'))

                        # Exclude .csv_bi files
                        csv_files = [f for f in csv_files if not f.name.endswith('.csv_bi')]

                        for csv_file in csv_files:
                            csv_info = self.extract_csv_labels(csv_file)
                            if csv_info:
                                self.file_info.append(csv_info)
                                montage_data['files'].append(csv_info)
                                patient_data['total_duration'] += csv_info['total_duration']
                                patient_data['seizure_types_found'].update(csv_info['labels'])
                                patient_data['csv_files'].append(str(csv_file))

                                # Update global statistics
                                for label, count in csv_info['label_counts'].items():
                                    self.label_stats[label] += count

                        for edf_file in edf_files:
                            patient_data['edf_files'].append(str(edf_file))

                        patient_data['total_files'] += len(csv_files)
                        session_data['montages'].append(montage_data)

                patient_data['sessions'].append(session_data)

        patient_data['seizure_types_found'] = list(patient_data['seizure_types_found'])

        if verbose:
            print(f"  📁 Sessions: {len(patient_data['sessions'])}")
            print(f"  📄 Total files: {patient_data['total_files']}")
            print(f"  ⏱️  Total duration: {patient_data['total_duration']:.1f} seconds")
            print(f"  🧠 Seizure types: {', '.join(patient_data['seizure_types_found'])}")

        self.patient_stats[patient_id] = patient_data
        return patient_data

    def merge_edf_files(self, edf_files, output_path=None):
        """
        Merge multiple EDF files

        Parameters:
        -----------
        edf_files : list
            List of EDF file paths
        output_path : str, optional
            Path untuk menyimpan file gabungan

        Returns:
        --------
        mne.io.Raw: Combined EEG data
        """
        try:
            print(f"🔄 Merging {len(edf_files)} EDF files...")

            raw_files = []
            for edf_file in edf_files:
                try:
                    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                    raw_files.append(raw)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {edf_file}: {e}")
                    continue

            if not raw_files:
                print("❌ No valid EDF files to merge")
                return None

            # Concatenate files
            if len(raw_files) == 1:
                combined_raw = raw_files[0]
            else:
                combined_raw = mne.concatenate_raws(raw_files, preload=False)

            print(f"✅ Successfully merged files")
            print(f"  📊 Channels: {len(combined_raw.ch_names)}")
            print(f"  ⏱️  Duration: {combined_raw.times[-1]:.1f} seconds")
            print(f"  📡 Sampling rate: {combined_raw.info['sfreq']} Hz")

            if output_path:
                combined_raw.save(output_path, overwrite=True)
                print(f"💾 Saved merged file to: {output_path}")

            return combined_raw

        except Exception as e:
            print(f"❌ Error merging EDF files: {e}")
            return None

    def generate_statistics_report(self):
        """Generate comprehensive statistics report"""
        print("\n📈 Generating Statistics Report...")

        report = {
            'dataset_summary': {
                'total_patients': len(self.patient_stats),
                'total_files': len(self.file_info),
                'total_duration_hours': sum(info['total_duration'] for info in self.file_info) / 3600,
                'seizure_types_found': list(self.label_stats.keys())
            },
            'label_distribution': dict(self.label_stats),
            'patient_summaries': {}
        }

        # Calculate per-patient summaries
        for patient_id, patient_data in self.patient_stats.items():
            report['patient_summaries'][patient_id] = {
                'sessions': len(patient_data['sessions']),
                'total_files': patient_data['total_files'],
                'duration_minutes': patient_data['total_duration'] / 60,
                'seizure_types': patient_data['seizure_types_found']
            }

        return report

    def plot_label_distribution(self, save_path=None):
        """Create visualization of label distribution"""
        try:
            plt.figure(figsize=(12, 8))

            # Prepare data for plotting
            labels = list(self.label_stats.keys())
            counts = list(self.label_stats.values())

            # Create readable labels
            readable_labels = [self.seizure_types.get(label, label) for label in labels]

            # Create bar plot
            plt.subplot(2, 1, 1)
            bars = plt.bar(readable_labels, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Distribution of Seizure Label Occurrences', fontsize=14, fontweight='bold')
            plt.xlabel('Seizure Type')
            plt.ylabel('Number of Occurrences')
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom')

            # Create pie chart
            plt.subplot(2, 1, 2)
            plt.pie(counts, labels=readable_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Seizure Label Distribution (Percentage)', fontsize=14, fontweight='bold')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"❌ Error creating plot: {e}")

    def export_to_csv(self, output_file):
        """Export analysis results to CSV"""
        try:
            # Create comprehensive DataFrame
            export_data = []

            for info in self.file_info:
                base_info = {
                    'file_path': info['file_path'],
                    'patient_id': Path(info['file_path']).parts[-4],
                    'session_id': Path(info['file_path']).parts[-3],
                    'montage_type': Path(info['file_path']).parts[-2],
                    'total_duration': info['total_duration'],
                    'num_channels': info['num_channels']
                }

                # Add label information
                for label in self.seizure_types.keys():
                    base_info[f'{label}_count'] = info['label_counts'].get(label, 0)
                    base_info[f'{label}_duration'] = info['label_durations'].get(label, 0)

                export_data.append(base_info)

            df = pd.DataFrame(export_data)
            df.to_csv(output_file, index=False)
            print(f"📄 Data exported to: {output_file}")

        except Exception as e:
            print(f"❌ Error exporting data: {e}")

    def export_labels_for_ml(self, output_dir):
        """
        Export processed labels in format suitable for machine learning

        Parameters:
        -----------
        output_dir : str
            Directory untuk menyimpan output files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Create time-series labels for each file
            for info in self.file_info:
                csv_file = info['file_path']

                # Read original CSV
                with open(csv_file, 'r') as f:
                    lines = f.readlines()

                header_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('channel,'):
                        header_idx = i
                        break

                df = pd.read_csv(csv_file, skiprows=header_idx)

                # Create output filename
                file_stem = Path(csv_file).stem
                output_file = output_path / f"{file_stem}_labels.csv"

                # Process and save
                df_processed = df[['channel', 'start_time', 'stop_time', 'label', 'confidence']].copy()
                df_processed['duration'] = df_processed['stop_time'] - df_processed['start_time']
                df_processed['seizure_type'] = df_processed['label'].map(self.seizure_types)

                df_processed.to_csv(output_file, index=False)

            print(f"🤖 ML-ready labels exported to: {output_dir}")

        except Exception as e:
            print(f"❌ Error exporting ML labels: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TUSZ Dataset Analyzer')
    parser.add_argument('--dataset_path', required=True,
                       help='Path ke direktori dataset TUSZ')
    parser.add_argument('--patient_id',
                       help='Analyze specific patient (optional)')
    parser.add_argument('--merge_edf',
                       help='Path ke direktori untuk merge EDF files')
    parser.add_argument('--output_dir', default='./tusz_analysis_output',
                       help='Directory untuk menyimpan hasil analisis')
    parser.add_argument('--export_csv', action='store_true',
                       help='Export results to CSV')
    parser.add_argument('--export_ml', action='store_true',
                       help='Export labels for machine learning')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    print("🧠 TUSZ Dataset Analyzer")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TUSZAnalyzer(args.dataset_path)

    # Scan dataset
    patients = analyzer.scan_dataset()

    if args.patient_id:
        # Analyze specific patient
        if args.patient_id in patients:
            analyzer.process_patient_data(args.patient_id, verbose=True)
        else:
            print(f"❌ Patient {args.patient_id} not found in dataset")
            return
    else:
        # Analyze all patients
        print(f"\n🔄 Processing {len(patients)} patients...")
        for i, patient_id in enumerate(patients):
            if args.verbose:
                print(f"\nProgress: {i+1}/{len(patients)}")
            analyzer.process_patient_data(patient_id, verbose=args.verbose)

    # Generate report
    report = analyzer.generate_statistics_report()

    # Print summary
    print("\n📊 DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Patients: {report['dataset_summary']['total_patients']}")
    print(f"Total Files: {report['dataset_summary']['total_files']}")
    print(f"Total Duration: {report['dataset_summary']['total_duration_hours']:.1f} hours")
    print(f"Seizure Types Found: {len(report['dataset_summary']['seizure_types_found'])}")

    print("\n🏷️  LABEL DISTRIBUTION")
    print("-" * 30)
    for label, count in report['label_distribution'].items():
        label_name = analyzer.seizure_types.get(label, label)
        print(f"{label} ({label_name}): {count:,}")

    # Save report
    report_file = output_path / 'analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Full report saved to: {report_file}")

    # Export to CSV if requested
    if args.export_csv:
        csv_file = output_path / 'dataset_analysis.csv'
        analyzer.export_to_csv(csv_file)

    # Export ML-ready labels if requested
    if args.export_ml:
        ml_dir = output_path / 'ml_labels'
        analyzer.export_labels_for_ml(ml_dir)

    # Create plots if requested
    if args.plot:
        plot_file = output_path / 'label_distribution.png'
        analyzer.plot_label_distribution(save_path=plot_file)

    # Merge EDF files if requested
    if args.merge_edf and args.patient_id:
        patient_data = analyzer.patient_stats.get(args.patient_id)
        if patient_data and patient_data['edf_files']:
            output_edf = Path(args.merge_edf) / f"{args.patient_id}_merged.fif"
            analyzer.merge_edf_files(patient_data['edf_files'], output_edf)

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
