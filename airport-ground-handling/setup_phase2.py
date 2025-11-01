#!/usr/bin/env python3
"""
Setup script for Phase 2 - RAG system

This script prepares everything needed for the RAG system
"""

import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    print("="*70)
    print("CHECKING REQUIREMENTS")
    print("="*70)
    
    required_packages = [
        'sentence-transformers',
        'chromadb',
        'transformers',
        'torch'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements_phase2.txt")
        return False
    
    print("\n✅ All requirements satisfied")
    return True


def check_data():
    """Check if data is available"""
    print("\n" + "="*70)
    print("CHECKING DATA")
    print("="*70)
    
    required_paths = [
        "data/raw/flight_schedules.csv",
        "data/raw/tasks.csv",
        "data/processed/scenarios"
    ]
    
    all_exist = True
    
    for path in required_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                print(f"✅ {path}")
            else:
                scenario_files = list(path_obj.glob("*.json"))
                print(f"✅ {path} ({len(scenario_files)} scenarios)")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Missing data files")
        print("\nGenerate data first:")
        print("  python generate_dataset.py --days 30")
        return False
    
    print("\n✅ All data files present")
    return True


def check_ollama():
    """Check if Ollama is available"""
    print("\n" + "="*70)
    print("CHECKING OLLAMA (Optional)")
    print("="*70)
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running")
            print(f"   Available models: {len(models)}")
            
            # Check for recommended model
            model_names = [m['name'] for m in models]
            if any('mistral' in name for name in model_names):
                print("✅ Mistral model available")
            else:
                print("⚠️  Mistral model not found")
                print("\nInstall with:")
                print("  ollama pull mistral:7b-instruct")
            
            return True
        
    except Exception as e:
        print("⚠️  Ollama not running")
        print("\nOllama is optional but recommended for better answers.")
        print("\nTo install:")
        print("  1. Visit: https://ollama.ai")
        print("  2. Download and install")
        print("  3. Run: ollama pull mistral:7b-instruct")
        print("\nThe system will work without Ollama but with limited generation.")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n" + "="*70)
    print("CREATING DIRECTORIES")
    print("="*70)
    
    directories = [
        "data/processed/embeddings",
        "data/processed/chroma_db",
        "models/embeddings",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print("\n✅ All directories created")


def main():
    """Main setup function"""
    
    print("\n")
    print("="*70)
    print("PHASE 2 SETUP - RAG SYSTEM")
    print("="*70)
    
    steps = [
        ("Requirements", check_requirements),
        ("Data", check_data),
        ("Ollama", check_ollama),
        ("Directories", create_directories),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    for step_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    critical_steps = ["Requirements", "Data"]
    all_critical_ok = all(results.get(step, False) for step in critical_steps)
    
    if all_critical_ok:
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE - Ready to build RAG system!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Build RAG system:")
        print("     python build_rag_system.py")
        print("\n  2. Run demos:")
        print("     python demo_rag.py")
        print("\n  3. Interactive mode:")
        print("     python demo_rag.py --mode interactive")
    else:
        print("\n" + "="*70)
        print("⚠️  SETUP INCOMPLETE")
        print("="*70)
        print("\nPlease fix the issues above before continuing.")
    
    print("\n")


if __name__ == "__main__":
    main()