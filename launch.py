#!/usr/bin/env python3
"""
FLUX Pipeline Launcher
Quick launcher for both dashboards
"""

import subprocess
import sys
import time
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required = [
        "flux_generator_dashboard.py",
        "analysis_dashboard.py",
        "interact_agent.py"
    ]

    missing = []
    for file in required:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        return False

    print("✅ All required files found")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║       FLUX.1-Kontext Interactive Pipeline Launcher      ║
╚══════════════════════════════════════════════════════════╝
""")

    if not check_files():
        print("\n⚠️  Please ensure all files are in the current directory")
        sys.exit(1)

    print("\nWhat would you like to do?\n")
    print("1. 🎨 Start Generator Dashboard (create new images)")
    print("2. 📊 Start Analysis Dashboard (review results)")
    print("3. 🚀 Start BOTH (separate terminals)")
    print("4. ❌ Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        print("\n🚀 Launching Generator Dashboard...")
        print("   URL will be: http://localhost:7860")
        print("   Press Ctrl+C to stop\n")
        subprocess.run([sys.executable, "flux_generator_dashboard.py"])

    elif choice == "2":
        print("\n🚀 Launching Analysis Dashboard...")
        print("   URL will be: http://localhost:7861")
        print("   Press Ctrl+C to stop\n")
        subprocess.run([sys.executable, "analysis_dashboard.py"])

    elif choice == "3":
        print("\n🚀 Launching both dashboards...")
        print("   Generator: http://localhost:7860")
        print("   Analysis:  http://localhost:7861")
        print("\n⚠️  Note: You'll need to open them in separate terminals")
        print("\nRun these commands in separate terminals:")
        print("   Terminal 1: python flux_generator_dashboard.py")
        print("   Terminal 2: python analysis_dashboard.py")

    elif choice == "4":
        print("\n👋 Goodbye!")
        sys.exit(0)

    else:
        print("\n❌ Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
        sys.exit(0)
