#!/usr/bin/env python3
"""Download full FlyWire connectome data.

This script downloads the complete FlyWire Drosophila connectome from CODEx API.

Download includes:
- synapses.csv: All synaptic connections (~50M edges)
- neurons.csv: Neuron metadata with cell types (~130k neurons)

Usage:
    python src/download_flywire.py

Requirements:
    - ~5GB disk space
    - Stable internet connection
    - 16GB+ RAM recommended for preprocessing
"""

import sys
import time
from pathlib import Path

# Try imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("requests not installed. Install with: pip install requests")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas not installed. Install with: pip install pandas")


PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def ensure_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_with_requests(url: str, output_path: Path) -> bool:
    """Download file using requests library with progress."""
    print(f"Downloading from: {url}")
    print(f"Output: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress update every 10%
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (total_size // 10) < chunk_size:
                            print(f"  Progress: {percent:.1f}% ({downloaded / 1e6:.1f} MB)")

        print(f"Download complete: {output_path}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_with_curl(url: str, output_path: Path) -> bool:
    """Download file using curl (more reliable for large files)."""
    import subprocess

    print(f"Downloading from: {url}")
    print(f"Output: {output_path}")

    try:
        cmd = ['curl', '-L', '-o', str(output_path), url]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            print(f"Download complete: {output_path}")
            return True
        else:
            print(f"curl failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Download timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print("curl not found. Try: brew install curl (macOS) or apt install curl (Linux)")
        return False


def get_curl_version() -> bool:
    """Check if curl is available."""
    import subprocess
    try:
        result = subprocess.run(['curl', '--version'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False


def main():
    """Main entry point."""
    ensure_dirs()

    print("=" * 60)
    print("FlyWire Connectome Data Download")
    print("=" * 60)
    print()

    # Data URLs
    synapses_url = "https://codex.flywire.ai/api/v1/table/flywire_neuropil_synapses"
    neurons_url = "https://codex.flywire.ai/api/v1/table/flywire_neuropil_neurons"

    # Output paths
    synapses_path = DATA_RAW / "synapses.csv"
    neurons_path = DATA_RAW / "neurons.csv"

    # Check what's already downloaded
    synapses_exists = synapses_path.exists()
    neurons_exists = neurons_path.exists()

    print("Download targets:")
    print(f"  Synapses: {synapses_path} ({'exists' if synapses_exists else 'pending'})")
    print(f"  Neurons: {neurons_path} ({'exists' if neurons_exists else 'pending'})")
    print()

    # Choose download method
    use_curl = get_curl_version()
    if use_curl:
        print("Using curl for download (recommended for large files)")
    elif HAS_REQUESTS:
        print("Using requests library for download")
    else:
        print("No download method available. Please install:")
        print("  pip install requests")
        print("  or install curl: brew install curl")
        return

    # Download synapses (large file)
    if not synapses_exists:
        print()
        print("-" * 60)
        print("Step 1: Downloading synapses (~2-4GB, may take 10-30 minutes)")
        print("-" * 60)

        if use_curl:
            success = download_with_curl(synapses_url, synapses_path)
        else:
            success = download_with_requests(synapses_url, synapses_path)

        if not success:
            print("\nSynapse download failed. You can retry later.")
            print("Partial download may be corrupted - remove and retry.")
    else:
        print("Synapses already downloaded, skipping...")

    # Download neurons (smaller file)
    if not neurons_exists:
        print()
        print("-" * 60)
        print("Step 2: Downloading neuron metadata (~50-100MB)")
        print("-" * 60)

        if use_curl:
            success = download_with_curl(neurons_url, neurons_path)
        else:
            success = download_with_requests(neurons_url, neurons_path)

        if not success:
            print("\nNeuron metadata download failed.")
    else:
        print("Neurons already downloaded, skipping...")

    # Summary
    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)

    if synapses_path.exists():
        size_mb = synapses_path.stat().st_size / 1e6
        print(f"  Synapses: {size_mb:.1f} MB")
    else:
        print("  Synapses: NOT DOWNLOADED")

    if neurons_path.exists():
        size_mb = neurons_path.stat().st_size / 1e6
        print(f"  Neurons: {size_mb:.1f} MB")
    else:
        print("  Neurons: NOT DOWNLOADED")

    print()

    # Next steps
    if synapses_path.exists():
        print("Next: Run preprocessing")
        print("  cd exp0418_brainSimplex")
        print("  python src/preprocessing.py")
    else:
        print("Please download synapses.csv first before running preprocessing.")

    print()


if __name__ == "__main__":
    main()
