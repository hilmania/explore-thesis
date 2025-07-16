#!/usr/bin/env python3
"""
TUSZ Dataset Analyzer - Setup Script
===================================

Script untuk menginstall dependencies yang dibutuhkan.

Usage:
    python setup_dependencies.py

Author: Assistant
Date: July 2025
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function"""
    print("🔧 TUSZ Dataset Analyzer - Setup")
    print("=" * 40)

    # List of required packages
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "mne>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0"
    ]

    print("📦 Installing required packages...")

    failed_packages = []

    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)

    if failed_packages:
        print(f"\n⚠️ Failed to install: {', '.join(failed_packages)}")
        print("Please install them manually using:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("\n✅ All packages installed successfully!")

    print("\n🚀 Setup complete! You can now run:")
    print("  python quick_analysis.py")
    print("  python tusz_analyzer.py --help")

if __name__ == "__main__":
    main()
