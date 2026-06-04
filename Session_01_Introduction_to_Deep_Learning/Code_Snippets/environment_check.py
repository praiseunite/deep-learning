"""
============================================================
  FILE: environment_check.py
  SESSION: 01 — Introduction to Deep Learning
  PURPOSE: Verify that all required libraries are installed
           and the Python environment is ready for this course.
  
  HOW TO RUN:
    1. Open Command Prompt (search for "cmd" in Windows)
    2. Type: python environment_check.py
    3. Press Enter
  
  EXPECTED OUTPUT:
    ✅ All libraries checked with version numbers
    🎉 Ready message if everything is installed
============================================================
"""

# We use 'importlib' to check if libraries exist without crashing
import importlib
import sys

# ─── Helper Function ───────────────────────────────────────────────────────────
def check_library(library_name, import_name=None):
    """
    Tries to import a library and reports whether it's installed.
    
    Parameters:
        library_name (str): The name shown to the user (e.g., "TensorFlow")
        import_name  (str): The actual import name if different (e.g., "tensorflow")
    
    Returns:
        bool: True if library is installed, False if not
    """
    # If no import_name given, use the library_name in lowercase
    if import_name is None:
        import_name = library_name.lower()
    
    try:
        # Try to import the library
        module = importlib.import_module(import_name)
        
        # Try to get the version number (most libraries have .__version__)
        version = getattr(module, '__version__', 'version unknown')
        
        # Print success message with green checkmark
        print(f"  ✅ {library_name:<20} OK  (version: {version})")
        return True
    
    except ImportError:
        # If import fails, print failure message with red X
        print(f"  ❌ {library_name:<20} NOT INSTALLED")
        print(f"     Fix: Open Command Prompt and run: pip install {import_name}")
        return False

# ─── Main Verification Script ─────────────────────────────────────────────────
def main():
    """
    Main function that runs all library checks and gives a final summary.
    """
    
    # Print header
    print("=" * 60)
    print("  🔍 Deep Learning Environment Verification")
    print("  Course: Deep Learning Using Neural Networks (Aptech)")
    print("  Session: 01 — Introduction to Deep Learning")
    print("=" * 60)
    print()
    
    # ── Check Python version first ─────────────────────────────────────────────
    print("📌 Checking Python version...")
    python_version = sys.version_info
    
    if python_version.major == 3 and python_version.minor >= 9:
        print(f"  ✅ Python               OK  (version: {sys.version.split()[0]})")
        python_ok = True
    else:
        print(f"  ❌ Python               OUTDATED (version: {sys.version.split()[0]})")
        print(f"     Fix: Download Python 3.10+ from https://www.python.org/downloads/")
        python_ok = False
    
    print()
    
    # ── Check Required Libraries ───────────────────────────────────────────────
    print("📌 Checking required libraries...")
    print()
    
    # Dictionary of {display_name: import_name}
    # We separate these because sometimes the pip name and import name differ
    libraries = {
        "NumPy":        "numpy",        # Numerical computing (arrays, math)
        "Pandas":       "pandas",       # Data manipulation (tables, CSVs)
        "Matplotlib":   "matplotlib",   # Plotting and visualization
        "Scikit-Learn": "sklearn",      # Classical ML algorithms
        "TensorFlow":   "tensorflow",   # Deep Learning framework by Google
        "Keras":        "keras",        # High-level DL API (now part of TensorFlow)
    }
    
    # Run the check for each library and collect results
    results = []
    for display_name, import_name in libraries.items():
        success = check_library(display_name, import_name)
        results.append(success)
    
    print()
    
    # ── Optional but Useful Libraries ─────────────────────────────────────────
    print("📌 Checking optional libraries (nice to have)...")
    print()
    
    optional_libraries = {
        "Seaborn":      "seaborn",      # Advanced visualization
        "Pillow (PIL)": "PIL",          # Image processing
        "OpenCV":       "cv2",          # Computer vision
    }
    
    for display_name, import_name in optional_libraries.items():
        check_library(display_name, import_name)
    
    print()
    
    # ── Final Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    
    all_ok = all(results) and python_ok
    
    if all_ok:
        print()
        print("  🎉 CONGRATULATIONS! Your environment is fully ready.")
        print("  🚀 You are all set for the Deep Learning course!")
        print()
        print("  Next Steps:")
        print("  1. Take a screenshot of this output")
        print("  2. Submit the screenshot with your Session 1 Assignment")
        print("  3. Come to Session 2 ready to dive deeper!")
        print()
    else:
        print()
        print("  ⚠️  Some libraries are missing or outdated.")
        print("  Please install the missing items listed above with ❌")
        print()
        print("  General Fix Command:")
        print("  pip install numpy pandas matplotlib scikit-learn tensorflow keras")
        print()
        print("  If you need help, ask your instructor before Session 2!")
        print()
    
    print("=" * 60)

# ─── Entry Point ───────────────────────────────────────────────────────────────
# This is the standard Python way to say "run main() when this file is executed"
# It will NOT run if this file is imported by another script
if __name__ == "__main__":
    main()
