import sys

print("-" * 50)
print("Python Version Verification")
print("-" * 50)
print("Python version:", sys.version)
print("")

# Test NumPy
try:
    import numpy as np
    print("[SUCCESS] NumPy version:", np.__version__)
except ImportError:
    print("[FAILED] NumPy not installed")

# Test OpenCV
try:
    import cv2
    print("[SUCCESS] OpenCV version:", cv2.__version__)
except ImportError:
    print("[FAILED] OpenCV not installed")

# Test Matplotlib
try:
    import matplotlib
    print("[SUCCESS] Matplotlib version:", matplotlib.__version__)
except ImportError:
    print("[FAILED] Matplotlib not installed")

# Test scikit-image
try:
    import skimage
    print("[SUCCESS] scikit-image version:", skimage.__version__)
except ImportError:
    print("[FAILED] scikit-image not installed")

# Test Mahotas
try:
    import mahotas
    print("[SUCCESS] Mahotas version:", mahotas.__version__)
except ImportError:
    print("[FAILED] Mahotas not installed")

# Test scikit-learn
try:
    import sklearn
    print("[SUCCESS] scikit-learn version:", sklearn.__version__)
except ImportError:
    print("[FAILED] scikit-learn not installed")

print("-" * 50)
print("Verification Complete!")