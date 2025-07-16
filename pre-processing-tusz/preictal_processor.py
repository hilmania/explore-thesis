#!/usr/bin/env python3
"""
TUSZ Pre-ictal Data Processor
============================

Script untuk memproses data EEG yang sudah dipilih untuk ekstraksi
segmen pre-ictal 20 menit sebelum onset seizure untuk training
model prediksi seizure.

Features:
- Extract 20-minute pre-ictal segments
- Extract interictal segments as negative examples
- Preprocessing EEG signals
- Generate balanced dataset for training

Author: Assistant
Date: July 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("⚠️  Warning: MNE not available. EDF processing will be limited.")

class PreictalDataProcessor:
    """Class untuk memproses data pre-ictal dari dataset TUSZ"""

    def __init__(self, prediction_dataset_path, preictal_duration=20,
                 interictal_duration=20, sampling_rate=250):
        """
        Initialize pre-ictal data processor

        Parameters:
        -----------
        prediction_dataset_path : str
            Path ke file prediction_dataset.json
        preictal_duration : int
            Durasi segmen pre-ictal dalam menit (default: 20)
        interictal_duration : int
            Durasi segmen interictal dalam menit (default: 20)
        sampling_rate : int
            Target sampling rate untuk output (default: 250 Hz)
        """
        self.prediction_dataset_path = Path(prediction_dataset_path)
        self.preictal_duration = preictal_duration * 60  # Convert to seconds
        self.interictal_duration = interictal_duration * 60
        self.target_sampling_rate = sampling_rate

        # Load prediction dataset
        with open(self.prediction_dataset_path) as f:
            self.prediction_dataset = json.load(f)

        # Results storage
        self.preictal_segments = []
        self.interictal_segments = []
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_preictal_segments': 0,
            'total_interictal_segments': 0,
            'errors': []
        }

        print("⚡ Pre-ictal Data Processor")
        print(f"📊 Pre-ictal duration: {preictal_duration} minutes")
        print(f"📊 Interictal duration: {interictal_duration} minutes")
        print(f"📡 Target sampling rate: {sampling_rate} Hz")
        print("-" * 50)

    def process_dataset(self, output_dir, max_files=None, verbose=True):
        """
        Process entire prediction dataset

        Parameters:
        -----------
        output_dir : str
            Directory untuk menyimpan hasil processing
        max_files : int, optional
            Maximum files to process (for testing)
        verbose : bool
            Print progress information
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        files = self.prediction_dataset['files']
        if max_files:
            files = files[:max_files]

        print(f"🔄 Processing {len(files)} files for pre-ictal extraction...")

        for i, file_info in enumerate(files):
            if verbose and (i + 1) % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(files)} files processed")

            try:
                self._process_single_file(file_info, verbose=verbose)
                self.processing_stats['successful_files'] += 1
            except Exception as e:
                error_msg = f"Error processing {file_info['file_path']}: {str(e)}"
                self.processing_stats['errors'].append(error_msg)
                self.processing_stats['failed_files'] += 1

                if verbose:
                    print(f"❌ {error_msg}")

            self.processing_stats['total_files'] += 1

        # Save processed data
        self._save_processed_data(output_path)

        # Print summary
        self._print_processing_summary()

        return self.processing_stats

    def _process_single_file(self, file_info, verbose=False):
        """Process single file untuk extract pre-ictal dan interictal segments"""
        csv_path = Path(file_info['file_path'])
        edf_path = csv_path.with_suffix('.edf')

        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        # Load EEG data
        if MNE_AVAILABLE:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            eeg_data = raw.get_data()
            original_sfreq = raw.info['sfreq']
            channel_names = raw.ch_names
        else:
            raise ImportError("MNE-Python required for EDF processing")

        # Get seizure events
        seizure_events = file_info.get('suitable_events', [])
        total_duration = file_info['total_duration']

        if verbose:
            print(f"🔍 Processing: {csv_path.name}")
            print(f"   Duration: {total_duration/60:.1f} min, {len(seizure_events)} seizures")

        # Extract pre-ictal segments
        for event in seizure_events:
            seizure_start = event['start_time']

            # Define pre-ictal period (20 minutes before seizure)
            preictal_start = seizure_start - self.preictal_duration
            preictal_end = seizure_start

            if preictal_start >= 0:  # Ensure we have enough data before seizure
                preictal_segment = self._extract_segment(
                    eeg_data, preictal_start, preictal_end,
                    original_sfreq, channel_names
                )

                segment_info = {
                    'file_path': str(csv_path),
                    'patient_id': csv_path.parts[-4],
                    'session_id': csv_path.parts[-3],
                    'seizure_type': event['label'],
                    'seizure_start_time': seizure_start,
                    'segment_start_time': preictal_start,
                    'segment_end_time': preictal_end,
                    'segment_type': 'preictal',
                    'duration': self.preictal_duration,
                    'channels': channel_names,
                    'sampling_rate': self.target_sampling_rate,
                    'data_shape': preictal_segment.shape
                }

                self.preictal_segments.append({
                    'info': segment_info,
                    'data': preictal_segment
                })

                self.processing_stats['total_preictal_segments'] += 1

        # Extract interictal segments (far from any seizure)
        interictal_segments = self._find_interictal_periods(seizure_events, total_duration)

        for start_time, end_time in interictal_segments:
            if end_time - start_time >= self.interictal_duration:
                # Extract segment from middle of interictal period
                segment_start = start_time + (end_time - start_time - self.interictal_duration) / 2
                segment_end = segment_start + self.interictal_duration

                interictal_segment = self._extract_segment(
                    eeg_data, segment_start, segment_end,
                    original_sfreq, channel_names
                )

                segment_info = {
                    'file_path': str(csv_path),
                    'patient_id': csv_path.parts[-4],
                    'session_id': csv_path.parts[-3],
                    'segment_start_time': segment_start,
                    'segment_end_time': segment_end,
                    'segment_type': 'interictal',
                    'duration': self.interictal_duration,
                    'channels': channel_names,
                    'sampling_rate': self.target_sampling_rate,
                    'data_shape': interictal_segment.shape
                }

                self.interictal_segments.append({
                    'info': segment_info,
                    'data': interictal_segment
                })

                self.processing_stats['total_interictal_segments'] += 1
                break  # Only take one interictal segment per file

    def _extract_segment(self, eeg_data, start_time, end_time, original_sfreq, channel_names):
        """Extract and preprocess EEG segment"""
        # Convert time to samples
        start_sample = int(start_time * original_sfreq)
        end_sample = int(end_time * original_sfreq)

        # Extract segment
        segment = eeg_data[:, start_sample:end_sample]

        # Resample if needed
        if original_sfreq != self.target_sampling_rate:
            if MNE_AVAILABLE:
                segment = mne.filter.resample(segment, down=original_sfreq/self.target_sampling_rate)

        # Basic preprocessing
        segment = self._preprocess_segment(segment)

        return segment

    def _preprocess_segment(self, segment):
        """Apply basic preprocessing to EEG segment"""
        # Remove DC offset
        segment = segment - np.mean(segment, axis=1, keepdims=True)

        # Apply basic filtering (1-50 Hz bandpass would be ideal with MNE)
        # For now, just ensure reasonable range
        segment = np.clip(segment, -1000, 1000)  # Clip extreme values

        # Normalize per channel
        std_vals = np.std(segment, axis=1, keepdims=True)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        segment = segment / std_vals

        return segment

    def _find_interictal_periods(self, seizure_events, total_duration):
        """Find periods far from seizures for interictal extraction"""
        if not seizure_events:
            # If no seizures, entire recording is interictal
            return [(0, total_duration)]

        interictal_periods = []

        # Sort events by start time
        events_sorted = sorted(seizure_events, key=lambda x: x['start_time'])

        # Check period before first seizure
        first_seizure_start = events_sorted[0]['start_time']
        if first_seizure_start > self.interictal_duration + 300:  # 5 min buffer
            interictal_periods.append((0, first_seizure_start - 300))

        # Check periods between seizures
        for i in range(len(events_sorted) - 1):
            current_end = events_sorted[i]['end_time']
            next_start = events_sorted[i + 1]['start_time']

            gap_duration = next_start - current_end
            if gap_duration > self.interictal_duration + 600:  # 10 min buffer (5 min each side)
                period_start = current_end + 300
                period_end = next_start - 300
                interictal_periods.append((period_start, period_end))

        # Check period after last seizure
        last_seizure_end = events_sorted[-1]['end_time']
        if total_duration - last_seizure_end > self.interictal_duration + 300:
            interictal_periods.append((last_seizure_end + 300, total_duration))

        return interictal_periods

    def _save_processed_data(self, output_path):
        """Save processed pre-ictal and interictal data"""
        # Save pre-ictal data
        if self.preictal_segments:
            preictal_data = {
                'segments': [s['info'] for s in self.preictal_segments],
                'metadata': {
                    'total_segments': len(self.preictal_segments),
                    'segment_duration': self.preictal_duration,
                    'sampling_rate': self.target_sampling_rate,
                    'segment_type': 'preictal'
                }
            }

            with open(output_path / 'preictal_segments.json', 'w') as f:
                json.dump(preictal_data, f, indent=2, default=str)

            # Save actual EEG data as numpy arrays
            preictal_arrays = np.array([s['data'] for s in self.preictal_segments])
            np.save(output_path / 'preictal_data.npy', preictal_arrays)

        # Save interictal data
        if self.interictal_segments:
            interictal_data = {
                'segments': [s['info'] for s in self.interictal_segments],
                'metadata': {
                    'total_segments': len(self.interictal_segments),
                    'segment_duration': self.interictal_duration,
                    'sampling_rate': self.target_sampling_rate,
                    'segment_type': 'interictal'
                }
            }

            with open(output_path / 'interictal_segments.json', 'w') as f:
                json.dump(interictal_data, f, indent=2, default=str)

            # Save actual EEG data as numpy arrays
            interictal_arrays = np.array([s['data'] for s in self.interictal_segments])
            np.save(output_path / 'interictal_data.npy', interictal_arrays)

        # Save processing statistics
        with open(output_path / 'processing_stats.json', 'w') as f:
            json.dump(self.processing_stats, f, indent=2, default=str)

        print(f"💾 Processed data saved to: {output_path}")

    def _print_processing_summary(self):
        """Print summary of processing results"""
        print(f"\n📊 PROCESSING SUMMARY")
        print("=" * 40)
        print(f"📄 Total files: {self.processing_stats['total_files']}")
        print(f"✅ Successful: {self.processing_stats['successful_files']}")
        print(f"❌ Failed: {self.processing_stats['failed_files']}")

        if self.processing_stats['total_files'] > 0:
            success_rate = self.processing_stats['successful_files'] / self.processing_stats['total_files'] * 100
            print(f"📈 Success rate: {success_rate:.1f}%")

        print(f"\n🧠 EXTRACTED SEGMENTS")
        print(f"⚡ Pre-ictal: {self.processing_stats['total_preictal_segments']}")
        print(f"🔄 Interictal: {self.processing_stats['total_interictal_segments']}")

        total_segments = (self.processing_stats['total_preictal_segments'] +
                         self.processing_stats['total_interictal_segments'])
        if total_segments > 0:
            preictal_ratio = self.processing_stats['total_preictal_segments'] / total_segments * 100
            print(f"📊 Pre-ictal ratio: {preictal_ratio:.1f}%")

        if self.processing_stats['errors']:
            print(f"\n❌ ERRORS ({len(self.processing_stats['errors'])}):")
            for error in self.processing_stats['errors'][:5]:  # Show first 5 errors
                print(f"   {error}")

def load_processed_data(data_dir):
    """
    Utility function to load processed pre-ictal and interictal data

    Parameters:
    -----------
    data_dir : str
        Directory containing processed data

    Returns:
    --------
    dict: Dictionary containing loaded data and metadata
    """
    data_path = Path(data_dir)

    result = {}

    # Load pre-ictal data
    preictal_json = data_path / 'preictal_segments.json'
    preictal_npy = data_path / 'preictal_data.npy'

    if preictal_json.exists() and preictal_npy.exists():
        with open(preictal_json) as f:
            preictal_info = json.load(f)
        preictal_data = np.load(preictal_npy)

        result['preictal'] = {
            'data': preictal_data,
            'info': preictal_info
        }

    # Load interictal data
    interictal_json = data_path / 'interictal_segments.json'
    interictal_npy = data_path / 'interictal_data.npy'

    if interictal_json.exists() and interictal_npy.exists():
        with open(interictal_json) as f:
            interictal_info = json.load(f)
        interictal_data = np.load(interictal_npy)

        result['interictal'] = {
            'data': interictal_data,
            'info': interictal_info
        }

    # Load processing stats
    stats_file = data_path / 'processing_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            result['processing_stats'] = json.load(f)

    return result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TUSZ Pre-ictal Data Processor')
    parser.add_argument('--prediction_dataset', required=True,
                       help='Path to prediction_dataset.json file')
    parser.add_argument('--output_dir', default='./preictal_data',
                       help='Output directory for processed data')
    parser.add_argument('--preictal_duration', type=int, default=20,
                       help='Pre-ictal segment duration in minutes (default: 20)')
    parser.add_argument('--interictal_duration', type=int, default=20,
                       help='Interictal segment duration in minutes (default: 20)')
    parser.add_argument('--sampling_rate', type=int, default=250,
                       help='Target sampling rate in Hz (default: 250)')
    parser.add_argument('--max_files', type=int,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    if not MNE_AVAILABLE:
        print("❌ MNE-Python is required for EDF processing!")
        print("💡 Install with: pip install mne")
        sys.exit(1)

    # Initialize processor
    processor = PreictalDataProcessor(
        prediction_dataset_path=args.prediction_dataset,
        preictal_duration=args.preictal_duration,
        interictal_duration=args.interictal_duration,
        sampling_rate=args.sampling_rate
    )

    # Process dataset
    stats = processor.process_dataset(
        output_dir=args.output_dir,
        max_files=args.max_files,
        verbose=args.verbose
    )

    print(f"\n🎯 PRE-ICTAL DATA EXTRACTION COMPLETE!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"⚡ Pre-ictal segments: {stats['total_preictal_segments']}")
    print(f"🔄 Interictal segments: {stats['total_interictal_segments']}")

    print(f"\n📚 GENERATED FILES:")
    print(f"   🧠 preictal_data.npy - Pre-ictal EEG segments")
    print(f"   📊 preictal_segments.json - Pre-ictal metadata")
    print(f"   🔄 interictal_data.npy - Interictal EEG segments")
    print(f"   📋 interictal_segments.json - Interictal metadata")
    print(f"   📈 processing_stats.json - Processing statistics")

    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Load data: data = load_processed_data('{args.output_dir}')")
    print(f"   2. Feature extraction from EEG segments")
    print(f"   3. Train seizure prediction model")
    print(f"   4. Evaluate model performance")

if __name__ == "__main__":
    main()
