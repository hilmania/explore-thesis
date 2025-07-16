#!/usr/bin/env python3
"""
TUSZ Seizure Prediction Dataset Selector
========================================

Script untuk memilih dataset TUSZ yang sesuai untuk prediksi seizure
dengan fase pre-ictal 20 menit. Script ini akan:

1. Filter file dengan durasi minimal 20 menit
2. Identifikasi file yang memiliki seizure events
3. Analisis timeline seizure untuk memastikan ada periode pre-ictal yang cukup
4. Generate dataset yang siap untuk training model prediksi

Kriteria Selection:
- Durasi recording ≥ 20 menit
- Memiliki seizure events (selain 'bckg')
- Ada periode pre-ictal minimal 20 menit sebelum seizure onset
- Kualitas signal yang baik (tidak terlalu banyak artifacts)

Author: Assistant
Date: July 2025
"""

import os
import sys
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import argparse

class SeizurePredictionDatasetSelector:
    """Class untuk memilih dataset yang sesuai untuk prediksi seizure"""

    def __init__(self, base_path, min_duration_minutes=20, preictal_duration_minutes=20):
        """
        Initialize dataset selector

        Parameters:
        -----------
        base_path : str
            Path ke direktori dataset TUSZ
        min_duration_minutes : int
            Durasi minimal recording dalam menit (default: 20)
        preictal_duration_minutes : int
            Durasi fase pre-ictal yang dibutuhkan dalam menit (default: 20)
        """
        self.base_path = Path(base_path)
        self.min_duration_seconds = min_duration_minutes * 60
        self.preictal_duration_seconds = preictal_duration_minutes * 60

        # Statistics
        self.suitable_files = []
        self.rejected_files = []
        self.patient_statistics = defaultdict(dict)

        # Seizure types (excluding background)
        self.seizure_types = {
            'cpsz': 'Complex Partial Seizure',
            'gnsz': 'Generalized Non-specific Seizure',
            'fnsz': 'Focal Non-specific Seizure',
            'tcsz': 'Tonic-Clonic Seizure',
            'absz': 'Absence Seizure',
            'mysz': 'Myoclonic Seizure',
            'tnsz': 'Tonic Seizure',
            'spsz': 'Simple Partial Seizure'
        }

        print(f"🎯 Seizure Prediction Dataset Selector")
        print(f"📏 Minimum duration: {min_duration_minutes} minutes")
        print(f"⏰ Pre-ictal period: {preictal_duration_minutes} minutes")
        print("-" * 50)

    def analyze_csv_file(self, csv_path):
        """
        Analyze CSV file for seizure prediction suitability

        Parameters:
        -----------
        csv_path : Path
            Path to CSV file

        Returns:
        --------
        dict: Analysis results with suitability assessment
        """
        try:
            # Read CSV file
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            # Extract metadata
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

            # Get total duration
            total_duration = float(metadata.get('duration', '0').split()[0])

            # Check minimum duration requirement
            if total_duration < self.min_duration_seconds:
                return {
                    'file_path': str(csv_path),
                    'total_duration': total_duration,
                    'suitable': False,
                    'rejection_reason': f'Duration too short: {total_duration/60:.1f} min < {self.min_duration_seconds/60} min',
                    'metadata': metadata
                }

            # Read annotation data
            df = pd.read_csv(csv_path, skiprows=header_idx)

            # Find seizure events (non-background labels)
            seizure_events = []
            seizure_labels = df[df['label'] != 'bckg']

            if seizure_labels.empty:
                return {
                    'file_path': str(csv_path),
                    'total_duration': total_duration,
                    'suitable': False,
                    'rejection_reason': 'No seizure events found (only background)',
                    'metadata': metadata,
                    'seizure_events': []
                }

            # Group consecutive seizure annotations by time and label
            seizure_events = self._extract_seizure_events(df)

            if not seizure_events:
                return {
                    'file_path': str(csv_path),
                    'total_duration': total_duration,
                    'suitable': False,
                    'rejection_reason': 'No distinct seizure events found',
                    'metadata': metadata,
                    'seizure_events': []
                }

            # Check pre-ictal period availability
            suitable_events = []
            for event in seizure_events:
                if event['start_time'] >= self.preictal_duration_seconds:
                    suitable_events.append(event)

            if not suitable_events:
                return {
                    'file_path': str(csv_path),
                    'total_duration': total_duration,
                    'suitable': False,
                    'rejection_reason': f'No seizure with sufficient pre-ictal period (need {self.preictal_duration_seconds/60} min)',
                    'metadata': metadata,
                    'seizure_events': seizure_events
                }

            # Calculate statistics
            seizure_stats = self._calculate_seizure_statistics(suitable_events, total_duration)

            return {
                'file_path': str(csv_path),
                'total_duration': total_duration,
                'suitable': True,
                'rejection_reason': None,
                'metadata': metadata,
                'seizure_events': seizure_events,
                'suitable_events': suitable_events,
                'seizure_statistics': seizure_stats,
                'num_channels': len(df['channel'].unique()),
                'quality_score': self._calculate_quality_score(df, seizure_events, total_duration)
            }

        except Exception as e:
            return {
                'file_path': str(csv_path),
                'suitable': False,
                'rejection_reason': f'Error processing file: {str(e)}',
                'error': str(e)
            }

    def _extract_seizure_events(self, df):
        """Extract distinct seizure events from annotations"""
        seizure_events = []

        # Group by label (excluding background)
        seizure_df = df[df['label'] != 'bckg'].copy()

        if seizure_df.empty:
            return seizure_events

        # Sort by start time
        seizure_df = seizure_df.sort_values('start_time')

        # Group consecutive annotations of same type
        for label in seizure_df['label'].unique():
            label_df = seizure_df[seizure_df['label'] == label].copy()

            if label_df.empty:
                continue

            # Find continuous events (group by time gaps)
            events = []
            current_event = {
                'label': label,
                'start_time': label_df.iloc[0]['start_time'],
                'end_time': label_df.iloc[0]['stop_time'],
                'channels': [label_df.iloc[0]['channel']]
            }

            for _, row in label_df.iloc[1:].iterrows():
                # If there's a small gap (< 5 seconds), consider it same event
                if row['start_time'] - current_event['end_time'] < 5.0:
                    current_event['end_time'] = max(current_event['end_time'], row['stop_time'])
                    if row['channel'] not in current_event['channels']:
                        current_event['channels'].append(row['channel'])
                else:
                    # New event
                    current_event['duration'] = current_event['end_time'] - current_event['start_time']
                    current_event['num_channels'] = len(current_event['channels'])
                    events.append(current_event)

                    current_event = {
                        'label': label,
                        'start_time': row['start_time'],
                        'end_time': row['stop_time'],
                        'channels': [row['channel']]
                    }

            # Add last event
            current_event['duration'] = current_event['end_time'] - current_event['start_time']
            current_event['num_channels'] = len(current_event['channels'])
            events.append(current_event)

            seizure_events.extend(events)

        # Sort by start time
        seizure_events.sort(key=lambda x: x['start_time'])

        return seizure_events

    def _calculate_seizure_statistics(self, events, total_duration):
        """Calculate statistics for seizure events"""
        if not events:
            return {}

        durations = [event['duration'] for event in events]
        intervals = []

        for i in range(len(events) - 1):
            interval = events[i+1]['start_time'] - events[i]['end_time']
            intervals.append(interval)

        stats = {
            'num_seizures': len(events),
            'total_seizure_duration': sum(durations),
            'mean_seizure_duration': np.mean(durations),
            'std_seizure_duration': np.std(durations),
            'min_seizure_duration': min(durations),
            'max_seizure_duration': max(durations),
            'seizure_types': list(set([event['label'] for event in events])),
            'seizure_density': len(events) / (total_duration / 3600)  # seizures per hour
        }

        if intervals:
            stats.update({
                'mean_interictal_interval': np.mean(intervals),
                'std_interictal_interval': np.std(intervals),
                'min_interictal_interval': min(intervals),
                'max_interictal_interval': max(intervals)
            })

        return stats

    def _calculate_quality_score(self, df, seizure_events, total_duration):
        """Calculate quality score for the recording"""
        score = 100  # Start with perfect score

        # Penalize very short recordings
        if total_duration < 1800:  # < 30 minutes
            score -= 20

        # Penalize recordings with too few seizures
        if len(seizure_events) < 1:
            score -= 50
        elif len(seizure_events) == 1:
            score -= 10

        # Penalize recordings with very short seizures
        short_seizures = [e for e in seizure_events if e['duration'] < 10]
        score -= len(short_seizures) * 5

        # Bonus for recordings with multiple seizure types
        seizure_types = set([e['label'] for e in seizure_events])
        if len(seizure_types) > 1:
            score += 10

        # Penalize recordings with too many channels having same label (might indicate artifacts)
        total_annotations = len(df)
        unique_channels = len(df['channel'].unique())
        if total_annotations / unique_channels > 50:  # Too many annotations per channel
            score -= 15

        return max(0, min(100, score))  # Clamp between 0-100

    def scan_dataset(self, max_files=None, verbose=True):
        """
        Scan entire dataset for suitable files

        Parameters:
        -----------
        max_files : int, optional
            Maximum number of files to process (for testing)
        verbose : bool
            Print progress information
        """
        print("🔍 Scanning dataset for seizure prediction suitability...")

        # Find all CSV files
        csv_files = []
        for csv_file in self.base_path.rglob('*.csv'):
            if not csv_file.name.endswith('.csv_bi'):
                csv_files.append(csv_file)

        if max_files:
            csv_files = csv_files[:max_files]

        print(f"📄 Found {len(csv_files)} CSV files to analyze")

        suitable_count = 0
        rejected_count = 0

        for i, csv_file in enumerate(csv_files):
            if verbose and (i + 1) % 100 == 0:
                print(f"📊 Progress: {i+1}/{len(csv_files)} files processed")

            analysis = self.analyze_csv_file(csv_file)

            if analysis['suitable']:
                self.suitable_files.append(analysis)
                suitable_count += 1

                if verbose and suitable_count <= 10:  # Show first 10 suitable files
                    duration_min = analysis['total_duration'] / 60
                    num_seizures = len(analysis['suitable_events'])
                    print(f"✅ Suitable: {csv_file.name} ({duration_min:.1f} min, {num_seizures} seizures)")
            else:
                self.rejected_files.append(analysis)
                rejected_count += 1

                if verbose and rejected_count <= 5:  # Show first 5 rejections
                    print(f"❌ Rejected: {csv_file.name} - {analysis['rejection_reason']}")

        print(f"\n📊 SCANNING RESULTS")
        print(f"✅ Suitable files: {suitable_count}")
        print(f"❌ Rejected files: {rejected_count}")
        print(f"📈 Success rate: {suitable_count/len(csv_files)*100:.1f}%")

    def generate_prediction_dataset(self, output_dir, quality_threshold=70):
        """
        Generate prediction-ready dataset from suitable files

        Parameters:
        -----------
        output_dir : str
            Directory to save prediction dataset
        quality_threshold : int
            Minimum quality score for inclusion (0-100)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Filter by quality
        high_quality_files = [f for f in self.suitable_files if f.get('quality_score', 0) >= quality_threshold]

        print(f"\n🎯 Generating prediction dataset...")
        print(f"📊 High quality files (≥{quality_threshold}): {len(high_quality_files)}")

        # Create prediction dataset metadata
        prediction_dataset = {
            'dataset_info': {
                'creation_date': datetime.now().isoformat(),
                'total_suitable_files': len(self.suitable_files),
                'high_quality_files': len(high_quality_files),
                'min_duration_minutes': self.min_duration_seconds / 60,
                'preictal_duration_minutes': self.preictal_duration_seconds / 60,
                'quality_threshold': quality_threshold
            },
            'files': high_quality_files
        }

        # Save complete dataset info
        with open(output_path / 'prediction_dataset.json', 'w') as f:
            json.dump(prediction_dataset, f, indent=2, default=str)

        # Create CSV summary for easy analysis
        self._create_csv_summary(high_quality_files, output_path / 'prediction_dataset_summary.csv')

        # Create training/validation splits
        self._create_train_val_splits(high_quality_files, output_path)

        # Generate statistics report
        self._generate_prediction_statistics(high_quality_files, output_path)

        print(f"💾 Prediction dataset saved to: {output_dir}")

        return prediction_dataset

    def _create_csv_summary(self, files, output_file):
        """Create CSV summary of prediction dataset"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'file_path', 'patient_id', 'session_id', 'duration_minutes',
                'num_seizures', 'seizure_types', 'total_seizure_duration',
                'quality_score', 'num_channels', 'suitable_for_prediction'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for file_info in files:
                # Extract patient and session info from path
                path_parts = Path(file_info['file_path']).parts
                patient_id = path_parts[-4] if len(path_parts) >= 4 else 'unknown'
                session_id = path_parts[-3] if len(path_parts) >= 3 else 'unknown'

                stats = file_info.get('seizure_statistics', {})

                row = {
                    'file_path': file_info['file_path'],
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'duration_minutes': file_info['total_duration'] / 60,
                    'num_seizures': len(file_info.get('suitable_events', [])),
                    'seizure_types': '; '.join(stats.get('seizure_types', [])),
                    'total_seizure_duration': stats.get('total_seizure_duration', 0),
                    'quality_score': file_info.get('quality_score', 0),
                    'num_channels': file_info.get('num_channels', 0),
                    'suitable_for_prediction': True
                }

                writer.writerow(row)

    def _create_train_val_splits(self, files, output_path):
        """Create train/validation splits by patient"""
        # Group files by patient
        patient_files = defaultdict(list)
        for file_info in files:
            path_parts = Path(file_info['file_path']).parts
            patient_id = path_parts[-4] if len(path_parts) >= 4 else 'unknown'
            patient_files[patient_id].append(file_info)

        patients = list(patient_files.keys())
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(patients)

        # 80/20 train/val split by patients
        split_idx = int(0.8 * len(patients))
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]

        train_files = []
        val_files = []

        for patient in train_patients:
            train_files.extend(patient_files[patient])

        for patient in val_patients:
            val_files.extend(patient_files[patient])

        # Save splits
        splits = {
            'train': {
                'patients': train_patients,
                'files': [f['file_path'] for f in train_files],
                'num_files': len(train_files)
            },
            'validation': {
                'patients': val_patients,
                'files': [f['file_path'] for f in val_files],
                'num_files': len(val_files)
            }
        }

        with open(output_path / 'train_val_splits.json', 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"📊 Train/Val splits created:")
        print(f"   Train: {len(train_patients)} patients, {len(train_files)} files")
        print(f"   Val: {len(val_patients)} patients, {len(val_files)} files")

    def _generate_prediction_statistics(self, files, output_path):
        """Generate detailed statistics for prediction dataset"""
        stats = {
            'overview': {
                'total_files': len(files),
                'total_duration_hours': sum(f['total_duration'] for f in files) / 3600,
                'unique_patients': len(set(Path(f['file_path']).parts[-4] for f in files)),
                'average_quality_score': np.mean([f.get('quality_score', 0) for f in files])
            },
            'seizure_analysis': {
                'total_seizures': sum(len(f.get('suitable_events', [])) for f in files),
                'seizure_types_distribution': {},
                'duration_statistics': {},
                'quality_distribution': {}
            }
        }

        # Analyze seizure types
        all_seizure_types = []
        all_durations = []
        all_quality_scores = []

        for file_info in files:
            events = file_info.get('suitable_events', [])
            for event in events:
                all_seizure_types.append(event['label'])
                all_durations.append(file_info['total_duration'] / 60)
            all_quality_scores.append(file_info.get('quality_score', 0))

        seizure_type_counts = Counter(all_seizure_types)
        stats['seizure_analysis']['seizure_types_distribution'] = dict(seizure_type_counts)

        if all_durations:
            stats['seizure_analysis']['duration_statistics'] = {
                'mean_duration_minutes': np.mean(all_durations),
                'std_duration_minutes': np.std(all_durations),
                'min_duration_minutes': min(all_durations),
                'max_duration_minutes': max(all_durations)
            }

        if all_quality_scores:
            stats['seizure_analysis']['quality_distribution'] = {
                'mean_quality': np.mean(all_quality_scores),
                'std_quality': np.std(all_quality_scores),
                'min_quality': min(all_quality_scores),
                'max_quality': max(all_quality_scores)
            }

        with open(output_path / 'prediction_dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        return stats

    def print_summary(self):
        """Print summary of dataset selection results"""
        print(f"\n📊 DATASET SELECTION SUMMARY")
        print("=" * 50)

        total_files = len(self.suitable_files) + len(self.rejected_files)
        print(f"📄 Total files analyzed: {total_files}")
        print(f"✅ Suitable for prediction: {len(self.suitable_files)} ({len(self.suitable_files)/total_files*100:.1f}%)")
        print(f"❌ Rejected: {len(self.rejected_files)} ({len(self.rejected_files)/total_files*100:.1f}%)")

        if self.suitable_files:
            durations = [f['total_duration']/60 for f in self.suitable_files]
            quality_scores = [f.get('quality_score', 0) for f in self.suitable_files]

            print(f"\n📈 SUITABLE FILES STATISTICS")
            print(f"⏱️  Duration: {np.mean(durations):.1f} ± {np.std(durations):.1f} minutes")
            print(f"🏆 Quality: {np.mean(quality_scores):.1f} ± {np.std(quality_scores):.1f} points")

            # Count seizure types
            all_seizure_types = []
            for f in self.suitable_files:
                events = f.get('suitable_events', [])
                for event in events:
                    all_seizure_types.append(event['label'])

            seizure_counts = Counter(all_seizure_types)
            print(f"\n🧠 SEIZURE TYPES IN PREDICTION DATASET")
            for seizure_type, count in seizure_counts.most_common():
                description = self.seizure_types.get(seizure_type, 'Unknown')
                print(f"   {seizure_type}: {count} events ({description})")

        if self.rejected_files:
            print(f"\n❌ REJECTION REASONS")
            rejection_reasons = Counter([f.get('rejection_reason', 'Unknown') for f in self.rejected_files])
            for reason, count in rejection_reasons.most_common():
                print(f"   {reason}: {count} files")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TUSZ Seizure Prediction Dataset Selector')
    parser.add_argument('--dataset_path', required=True,
                       help='Path to TUSZ dataset directory')
    parser.add_argument('--min_duration', type=int, default=20,
                       help='Minimum recording duration in minutes (default: 20)')
    parser.add_argument('--preictal_duration', type=int, default=20,
                       help='Required pre-ictal period in minutes (default: 20)')
    parser.add_argument('--output_dir', default='./prediction_dataset',
                       help='Output directory for prediction dataset')
    parser.add_argument('--quality_threshold', type=int, default=70,
                       help='Minimum quality score for inclusion (default: 70)')
    parser.add_argument('--max_files', type=int,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Initialize selector
    selector = SeizurePredictionDatasetSelector(
        base_path=args.dataset_path,
        min_duration_minutes=args.min_duration,
        preictal_duration_minutes=args.preictal_duration
    )

    # Scan dataset
    selector.scan_dataset(max_files=args.max_files, verbose=args.verbose)

    # Print summary
    selector.print_summary()

    # Generate prediction dataset
    if selector.suitable_files:
        prediction_dataset = selector.generate_prediction_dataset(
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold
        )

        print(f"\n🎯 PREDICTION DATASET READY!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📄 Files included: {len(prediction_dataset['files'])}")
        print(f"🏆 Quality threshold: {args.quality_threshold}")

        print(f"\n📚 GENERATED FILES:")
        print(f"   📊 prediction_dataset.json - Complete dataset metadata")
        print(f"   📋 prediction_dataset_summary.csv - Summary table")
        print(f"   🔄 train_val_splits.json - Train/validation splits")
        print(f"   📈 prediction_dataset_statistics.json - Detailed statistics")

        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Load EDF files corresponding to selected CSV files")
        print(f"   2. Extract 20-minute pre-ictal segments before seizure onset")
        print(f"   3. Extract interictal segments for negative examples")
        print(f"   4. Preprocess EEG signals (filtering, normalization)")
        print(f"   5. Train seizure prediction model")
    else:
        print(f"\n❌ No suitable files found for prediction!")
        print(f"💡 Try reducing minimum duration or pre-ictal requirements")

if __name__ == "__main__":
    main()
