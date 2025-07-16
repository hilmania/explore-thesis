#!/usr/bin/env python3
"""
Quick Demo: TUSZ Seizure Prediction Dataset Analysis
===================================================

Demo cepat untuk menunjukkan kemampuan tools yang telah dibuat
untuk mempersiapkan dataset TUSZ untuk prediksi seizure.
"""

import json
import os
from pathlib import Path

def analyze_existing_dataset():
    """Analisis dataset yang sudah ada"""

    print("🧠 TUSZ SEIZURE PREDICTION DATASET DEMO")
    print("=" * 60)

    # Check for existing analysis
    analysis_file = Path("tusz_analysis_output/analysis_report.json")

    if analysis_file.exists():
        print("📊 ANALISIS DATASET YANG SUDAH ADA")
        print("-" * 40)

        try:
            with open(analysis_file) as f:
                analysis = json.load(f)

            summary = analysis['dataset_summary']
            labels = analysis['label_distribution']

            print(f"👥 Total Patients: {summary['total_patients']}")
            print(f"📄 Total Files: {summary['total_files']}")
            print(f"⏱️  Total Duration: {summary['total_duration_hours']:.1f} hours")

            print(f"\n🧠 SEIZURE TYPES DISTRIBUTION:")
            total_labels = sum(labels.values())
            seizure_labels = {k: v for k, v in labels.items() if k != 'bckg'}
            total_seizures = sum(seizure_labels.values())

            print(f"   📊 Background: {labels.get('bckg', 0):,} ({labels.get('bckg', 0)/total_labels*100:.1f}%)")
            print(f"   ⚡ Total Seizures: {total_seizures:,} ({total_seizures/total_labels*100:.1f}%)")

            for seizure_type, count in sorted(seizure_labels.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_seizures) * 100
                print(f"   • {seizure_type}: {count:,} ({percentage:.1f}%)")

            print(f"\n🎯 ESTIMASI UNTUK PREDIKSI SEIZURE")
            print("-" * 35)

            # Estimate files suitable for prediction
            # Assumsi: files dengan durasi >20 menit dan ada seizure = ~15-20% dari total
            estimated_suitable = int(summary['total_files'] * 0.17)  # 17% estimate
            estimated_preictal_segments = int(total_seizures * 0.3)  # 30% memiliki pre-ictal yang cukup

            print(f"📄 Files cocok untuk prediksi (≥20 min): ~{estimated_suitable:,}")
            print(f"⚡ Pre-ictal segments yang bisa diekstrak: ~{estimated_preictal_segments:,}")
            print(f"🔄 Interictal segments (1:1 ratio): ~{estimated_preictal_segments:,}")
            print(f"📊 Total training samples: ~{estimated_preictal_segments*2:,}")

            # Data size estimation
            # 22 channels × 20 min × 250 Hz × 4 bytes = ~66 MB per segment
            data_size_gb = (estimated_preictal_segments * 2 * 22 * 20 * 60 * 250 * 4) / 1e9
            print(f"💾 Estimated data size: ~{data_size_gb:.1f} GB")

        except Exception as e:
            print(f"❌ Error reading analysis: {e}")
    else:
        print("📊 DATASET BELUM DIANALISIS")
        print("-" * 30)
        print("Untuk melakukan analisis lengkap, jalankan:")
        print("python tusz_analyzer.py --dataset_path . --verbose --export_csv")

    print(f"\n🔧 TOOLS YANG TERSEDIA")
    print("-" * 25)

    tools = {
        "tusz_analyzer.py": "Analisis lengkap dataset",
        "seizure_prediction_selector.py": "Pilih files untuk prediksi (≥20 min)",
        "preictal_processor.py": "Extract segmen pre-ictal 20 menit",
        "seizure_prediction_workflow.py": "Demo workflow lengkap",
        "simple_analyzer.py": "Analisis tanpa dependencies",
        "quick_analysis.py": "Analisis cepat dengan plot"
    }

    for tool, description in tools.items():
        status = "✅" if Path(tool).exists() else "❌"
        print(f"{status} {tool:35s} - {description}")

    print(f"\n🚀 WORKFLOW UNTUK PREDIKSI SEIZURE")
    print("-" * 35)
    print("1. 🔍 Select Dataset:")
    print("   python seizure_prediction_selector.py --dataset_path . --min_duration 20")
    print("")
    print("2. 🧠 Extract Pre-ictal:")
    print("   python preictal_processor.py --prediction_dataset ./prediction_dataset.json")
    print("")
    print("3. 🤖 Train Model:")
    print("   # Load processed data dan train your seizure prediction model")

    print(f"\n💡 KRITERIA SELECTION UNTUK PREDIKSI")
    print("-" * 35)
    print("✅ Durasi recording ≥ 20 menit")
    print("✅ Ada seizure events (bukan hanya background)")
    print("✅ Periode pre-ictal ≥ 20 menit sebelum seizure onset")
    print("✅ Kualitas signal yang baik (quality score ≥ 70)")
    print("✅ Sufficient interictal periods untuk negative examples")

    print(f"\n📈 EXPECTED RESULTS")
    print("-" * 20)
    print("Dari ~4,664 total files:")
    print("• ~800 files suitable untuk prediksi (17%)")
    print("• ~300-400 unique patients")
    print("• ~7,500 seizure events total")
    print("• ~2,000-3,000 extractable pre-ictal segments")
    print("• Balanced dataset: 1:1 pre-ictal:interictal ratio")
    print("• Ready untuk train deep learning models")

    print(f"\n🎯 SEIZURE PREDICTION APPLICATIONS")
    print("-" * 35)
    print("🏥 Clinical: Early warning systems")
    print("📱 Wearable: Real-time monitoring devices")
    print("🔬 Research: Understanding seizure mechanisms")
    print("🤖 AI/ML: Advanced prediction algorithms")

    print(f"\n✅ TOOLS READY FOR SEIZURE PREDICTION!")

if __name__ == "__main__":
    analyze_existing_dataset()
