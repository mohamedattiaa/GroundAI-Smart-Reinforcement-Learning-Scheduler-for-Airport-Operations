"""
Verify Phase 2 installation
"""

def check_import(module_name, display_name=None):
    """Check if module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        module = __import__(module_name.replace('-', '_'))
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {display_name}: Not installed - {e}")
        return False


def main():
    print("="*60)
    print("VERIFYING PHASE 2 INSTALLATION")
    print("="*60)
    
    packages = [
        ('sentence_transformers', 'sentence-transformers'),
        ('transformers', 'transformers'),
        ('tensorflow', 'tensorflow'),
        ('chromadb', 'chromadb'),
        ('langchain', 'langchain'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
    ]
    
    success = 0
    total = len(packages)
    
    for module, display in packages:
        if check_import(module, display):
            success += 1
    
    print("\n" + "="*60)
    print(f"RESULT: {success}/{total} packages available")
    print("="*60)
    
    if success == total:
        print("\n✅ All packages installed correctly!")
        print("\nYou can now run:")
        print("  python build_rag_system.py")
    else:
        print("\n⚠️  Some packages missing. Run:")
        print("  python fix_dependencies.py")


if __name__ == "__main__":
    main()