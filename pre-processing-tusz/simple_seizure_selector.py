#!/usr/bin/env python3
"""
Simple seizure prediction dataset selector
Output hasil langsung ke file untuk memastikan kita bisa melihat hasilnya
"""

import os
import glob

def simple_seizure_analysis():
    """Analisis sederhana untuk mencari file yang cocok untuk prediksi seizure"""

    # Write results to file
    with open('simple_analysis_results.txt', 'w') as f:
        f.write("TUSZ Seizure Prediction Dataset Analysis\n")
        f.write("="*50 + "\n\n")

        # Find CSV files
        csv_files = glob.glob("*/*/*/*.csv")
        csv_files = [file for file in csv_files if not file.endswith('.csv_bi')]

        f.write(f"Found {len(csv_files)} CSV files\n")
        f.write(f"Sample files:\n")

        suitable_count = 0
        rejected_count = 0

        # Analyze first 20 files as sample
        sample_files = csv_files[:20]

        for i, csv_file in enumerate(sample_files):
            f.write(f"\n{i+1}. {csv_file}\n")

            try:
                # Read file
                with open(csv_file, 'r') as csv_f:
                    content = csv_f.read()

                lines = content.split('\n')

                # Extract duration
                duration_seconds = 0
                for line in lines:
                    if line.startswith('# duration'):
                        try:
                            duration_str = line.split('=')[1].strip()
                            duration_seconds = float(duration_str.split()[0])
                            break
                        except:
                            pass

                # Count seizure annotations
                seizure_count = 0
                seizure_types = set()
                for line in lines:
                    if 'seiz' in line and not line.startswith('#') and ',' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) >= 4:
                                label = parts[3].strip()
                                if label != 'bckg':
                                    seizure_count += 1
                                    seizure_types.add(label)
                        except:
                            pass

                # Analysis
                duration_minutes = duration_seconds / 60
                f.write(f"   Duration: {duration_minutes:.1f} minutes\n")
                f.write(f"   Seizure annotations: {seizure_count}\n")
                f.write(f"   Seizure types: {list(seizure_types)}\n")

                # Check criteria
                suitable = True
                reasons = []

                if duration_seconds < 1200:  # < 20 minutes
                    suitable = False
                    reasons.append(f"Too short ({duration_minutes:.1f} min < 20 min)")

                if seizure_count == 0:
                    suitable = False
                    reasons.append("No seizures")

                if suitable:
                    f.write("   ✅ SUITABLE for prediction\n")
                    suitable_count += 1
                else:
                    f.write(f"   ❌ NOT suitable: {', '.join(reasons)}\n")
                    rejected_count += 1

            except Exception as e:
                f.write(f"   ❌ Error: {e}\n")
                rejected_count += 1

        # Summary
        f.write(f"\n" + "="*50 + "\n")
        f.write(f"ANALYSIS SUMMARY (sample of {len(sample_files)} files):\n")
        f.write(f"✅ Suitable files: {suitable_count}\n")
        f.write(f"❌ Rejected files: {rejected_count}\n")
        f.write(f"📊 Selection rate: {suitable_count/len(sample_files)*100:.1f}%\n")
        f.write(f"🎯 Estimated suitable in full dataset: {int(suitable_count/len(sample_files)*len(csv_files))}\n")

        f.write(f"\nCRITERIA FOR SEIZURE PREDICTION:\n")
        f.write(f"1. Recording duration ≥ 20 minutes\n")
        f.write(f"2. Contains seizure events (not just background)\n")
        f.write(f"3. First seizure ≥ 20 minutes from start (for pre-ictal period)\n")

        f.write(f"\nNEXT STEPS:\n")
        f.write(f"1. Load corresponding EDF files for suitable recordings\n")
        f.write(f"2. Extract 20-minute pre-ictal segments before seizure onset\n")
        f.write(f"3. Extract interictal segments for negative examples\n")
        f.write(f"4. Preprocess EEG signals (filtering, artifact removal)\n")
        f.write(f"5. Train machine learning model for seizure prediction\n")

    print("Analysis complete! Results saved to simple_analysis_results.txt")
    return suitable_count, rejected_count

if __name__ == "__main__":
    suitable, rejected = simple_seizure_analysis()
    print(f"Found {suitable} suitable files, {rejected} rejected files")
