"""
check_cuda_setup.py
Verifies that your CUDA development environment is properly configured.
Run this to diagnose setup issues.
"""

import sys
import subprocess
from pathlib import Path


def check_python():
    """Check Python version"""
    print("=" * 60)
    print("PYTHON VERSION")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 8 <= version.minor <= 11:
        print("✓ Python version is compatible")
        return True
    else:
        print(
            "⚠ Python 3.8-3.11 recommended (you have {}.{})".format(
                version.major, version.minor
            )
        )
        return False


def check_numpy():
    """Check if NumPy is installed"""
    print("\n" + "=" * 60)
    print("NUMPY")
    print("=" * 60)
    try:
        import numpy as np

        print(f"✓ NumPy {np.__version__} installed")
        return True
    except ImportError:
        print("✗ NumPy not installed")
        print("  Fix: pip install numpy")
        return False


def check_pycuda():
    """Check if PyCUDA is installed"""
    print("\n" + "=" * 60)
    print("PYCUDA")
    print("=" * 60)
    try:
        import pycuda

        print(f"✓ PyCUDA {pycuda.VERSION_TEXT} installed")
        return True
    except ImportError:
        print("✗ PyCUDA not installed")
        print("  Fix: pip install pycuda")
        return False


def check_opencv():
    """Check if OpenCV is installed"""
    print("\n" + "=" * 60)
    print("OPENCV")
    print("=" * 60)
    try:
        import cv2

        print(f"✓ OpenCV {cv2.__version__} installed")
        return True
    except ImportError:
        print("✗ OpenCV not installed")
        print("  Fix: pip install opencv-python")
        return False


def check_pygame():
    """Check if PyGame is installed"""
    print("\n" + "=" * 60)
    print("PYGAME")
    print("=" * 60)
    try:
        import pygame

        print(f"✓ PyGame {pygame.version.ver} installed")
        return True
    except ImportError:
        print("✗ PyGame not installed")
        print("  Fix: pip install pygame")
        return False


def check_nvidia_smi():
    """Check if nvidia-smi is available"""
    print("\n" + "=" * 60)
    print("NVIDIA GPU DRIVER")
    print("=" * 60)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 3:
                    print(f"✓ GPU: {parts[0].strip()}")
                    print(f"  Driver: {parts[1].strip()}")
                    print(f"  Memory: {parts[2].strip()}")
            return True
        else:
            print("✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  Fix: Install/update NVIDIA drivers")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {e}")
        return False


def check_nvcc():
    """Check if NVCC is available"""
    print("\n" + "=" * 60)
    print("CUDA TOOLKIT")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    print(f"✓ NVCC: {line.strip()}")
                    return True
        print("✗ nvcc found but version check failed")
        return False
    except FileNotFoundError:
        print("✗ nvcc not found in PATH")
        print("  Fix: Install CUDA Toolkit and add to PATH")
        return False
    except Exception as e:
        print(f"✗ Error running nvcc: {e}")
        return False


def check_cl_exe():
    """Check if Visual Studio cl.exe is available"""
    print("\n" + "=" * 60)
    print("VISUAL STUDIO COMPILER")
    print("=" * 60)
    try:
        result = subprocess.run(["cl.exe"], capture_output=True, text=True, timeout=5)
        # cl.exe returns error code when called without args, but still prints version
        if "Microsoft (R) C/C++ Optimizing Compiler" in result.stderr:
            for line in result.stderr.split("\n"):
                if "Compiler Version" in line:
                    print(f"✓ {line.strip()}")
                    return True
        print("✗ cl.exe found but version check failed")
        return False
    except FileNotFoundError:
        print("✗ cl.exe not found in PATH")
        print("  Fix (Windows): Run `. .\\setup_cuda_env.ps1` or use run_with_msvc.ps1")
        print("  Or open 'Developer Command Prompt for VS 2022'")
        return False
    except Exception as e:
        print(f"✗ Error running cl.exe: {e}")
        return False


def check_cuda_device():
    """Check if CUDA device is accessible"""
    print("\n" + "=" * 60)
    print("CUDA DEVICE ACCESS")
    print("=" * 60)
    try:
        import pycuda.driver as cuda

        cuda.init()
        device_count = cuda.Device.count()

        if device_count == 0:
            print("✗ No CUDA devices found")
            return False

        print(f"✓ Found {device_count} CUDA device(s):")
        for i in range(device_count):
            device = cuda.Device(i)
            print(f"  [{i}] {device.name()}")
            attrs = device.get_attributes()
            compute_capability = (
                device.compute_capability()[0],
                device.compute_capability()[1],
            )
            print(
                f"      Compute Capability: {compute_capability[0]}.{compute_capability[1]}"
            )
            total_mem = device.total_memory() // (1024**2)
            print(f"      Memory: {total_mem} MB")
        return True

    except Exception as e:
        print(f"✗ Cannot access CUDA device: {e}")
        print("  This might be OK if cl.exe isn't in PATH yet")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("CUDA DEVELOPMENT ENVIRONMENT CHECK")
    print("=" * 60)

    results = {
        "Python": check_python(),
        "NumPy": check_numpy(),
        "PyCUDA": check_pycuda(),
        "OpenCV": check_opencv(),
        "PyGame": check_pygame(),
        "NVIDIA Driver": check_nvidia_smi(),
        "CUDA Toolkit": check_nvcc(),
        "VS Compiler": check_cl_exe(),
        "CUDA Device": check_cuda_device(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for component, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {component}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 Your CUDA development environment is fully configured!")
        print("   You can run: python hello_cuda.py")
    else:
        print("\n⚠ Some components need attention. See above for fixes.")
        if not results["VS Compiler"]:
            print("\n💡 On Windows, run this first:")
            print("   . .\\setup_cuda_env.ps1")
            print("   or use: .\\run_with_msvc.ps1 hello_cuda.py")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
