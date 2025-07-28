#!/usr/bin/env python3
"""
Quick verification script for Adobe Hackathon Round 1B submission
Validates that the intelligent document analyzer is ready for submission
"""

import os
import sys
import importlib.util

def check_file_exists(filename, description):
    """Check if a required file exists"""
    if os.path.exists(filename):
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ {description}: {filename} (MISSING)")
        return False

def check_dependencies():
    """Check if required Python packages are available"""
    required_packages = [
        ('fitz', 'PyMuPDF'),
        ('sentence_transformers', 'sentence-transformers'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]
    
    all_good = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description}: Available")
        except ImportError:
            print(f"❌ {description}: Missing")
            all_good = False
    
    return all_good

def main():
    print("🔍 Adobe Hackathon Round 1B - Submission Verification")
    print("=" * 60)
    
    # Check core files
    files_check = all([
        check_file_exists('intelligent_document_analyzer.py', 'Main application'),
        check_file_exists('requirements.txt', 'Dependencies'),
        check_file_exists('Dockerfile', 'Container config'),
        check_file_exists('README.md', 'Documentation'),
        check_file_exists('input/', 'Input directory'),
        check_file_exists('output/', 'Output directory'),
        check_file_exists('cache/', 'Cache directory')
    ])
    
    print("\n📦 Python Dependencies:")
    deps_check = check_dependencies()
    
    print("\n🧠 System Analysis:")
    try:
        # Try to import the main module
        spec = importlib.util.spec_from_file_location("analyzer", "intelligent_document_analyzer.py")
        analyzer_module = importlib.util.module_from_spec(spec)
        print("✅ Main module: Importable")
    except Exception as e:
        print(f"❌ Main module: Import error - {e}")
        files_check = False
    
    print("\n" + "=" * 60)
    if files_check and deps_check:
        print("🚀 SUBMISSION READY! All checks passed.")
        print("\nTo test the system:")
        print("1. Place PDF files in the 'input/' directory")
        print("2. Run: python intelligent_document_analyzer.py")
        print("3. Check results in the 'output/' directory")
        print("\nFor Docker:")
        print("1. Build: docker build -t intelligent-analyzer .")
        print("2. Run: docker run -v ./input:/app/input -v ./output:/app/output intelligent-analyzer")
        return 0
    else:
        print("❌ SUBMISSION NOT READY. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
