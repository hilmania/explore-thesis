#!/usr/bin/env python3
"""
Test TUSZ Dataset - Single Patient Analysis
===========================================

Test script untuk menganalisis satu patient saja untuk memastikan
script berfungsi dengan benar.
"""

import os
from pathlib import Path
import csv
from collections import Counter

def test_single_patient():
    """Test analysis for single patient"""

    # Path to dataset
    base_path = Path("/Volumes/Hilmania/TUH SZ/v2.0.3/edf/train")

    print("🧠 Testing TUSZ Analysis - Single Patient")
    print("=" * 50)

    # Find first patient
    patients = []
    for item in base_path.iterdir():
        if item.is_dir() and len(item.name) == 8:
            patients.append(item.name)

    if not patients:
        print("❌ No patients found!")
        return

    patient_id = patients[0]
    patient_path = base_path / patient_id

    print(f"👤 Testing with patient: {patient_id}")
    print(f"📂 Patient path: {patient_path}")

    # Find CSV files
    csv_files = []
    for root, dirs, files in os.walk(patient_path):
        for file in files:
            if file.endswith('.csv') and not file.endswith('.csv_bi'):
                csv_files.append(Path(root) / file)

    print(f"📄 Found {len(csv_files)} CSV files")

    if not csv_files:
        print("❌ No CSV files found!")
        return

    # Analyze first CSV file
    csv_file = csv_files[0]
    print(f"🔍 Analyzing: {csv_file.name}")

    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        print(f"📝 File has {len(lines)} lines")

        # Show first few lines
        print("\n📋 First 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1:2d}: {line.strip()}")

        # Find header
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('channel,'):
                header_idx = i
                print(f"\n🔍 Found header at line {i+1}")
                break

        # Parse data
        labels = []
        data_lines = lines[header_idx+1:]

        for line in data_lines[:5]:  # Just first 5 data lines
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    labels.append(parts[3])
                    print(f"📊 Data: {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]}")

        # Count labels
        if labels:
            label_counts = Counter(labels)
            print(f"\n🏷️  Labels found in sample: {dict(label_counts)}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_patient()
