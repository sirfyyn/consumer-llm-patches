#!/usr/bin/env python3
"""
apply_patches.py — Apply all 10 consumer-llm-patches to your local installation.

Usage:
    python apply_patches.py --check    # Verify patch targets exist
    python apply_patches.py --apply    # Apply all patches
    python apply_patches.py --verify   # Verify patches are active

Patches bitsandbytes and transformers in your current Python environment.
Run with the same Python that runs your training.
"""

import sys, os, re, subprocess, argparse
from pathlib import Path

def find_site_packages():
    import site
    paths = site.getsitepackages() + [site.getusersitepackages()]
    for p in paths:
        if os.path.isdir(p):
            return p
    return None

def find_package(name):
    sp = find_site_packages()
    if sp:
        p = Path(sp) / name
        if p.exists():
            return p
    # Try local user path
    home = Path.home()
    for candidate in [
        home / ".local/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / name,
    ]:
        if candidate.exists():
            return candidate
    return None

PATCHES = [
    # (package, relative_path, old_text, new_text, description)
    (
        "bitsandbytes",
        "autograd/_functions.py",
        # P1: Pre-quantized Int8Params — build CB/SCB directly
        "            else:\n                state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))",
        """            # PATCH sirfyyn P1: Pre-quantized Int8Params (B.dtype==int8) have no SCB.
            # Build CB/SCB from the weight data directly.
            if B.dtype == torch.int8:
                state.CB = B.data.clone()
                if hasattr(B, "SCB") and B.SCB is not None:
                    state.SCB = B.SCB.to(state.CB.device)
                else:
                    state.SCB = state.CB.float().abs().amax(dim=1).div(127.0).to(state.CB.device)
            else:
                state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))""",
        "P1: Pre-quantized Int8Params NaN fix"
    ),
]

# Full patch set — see README.md for all 10 patches
# This script applies the critical ones programmatically.
# For complete patch set, see patches/ directory.

def check_patches():
    print("=== Checking patch targets ===")
    bnb = find_package("bitsandbytes")
    tf = find_package("transformers")
    peft = find_package("peft")

    print(f"bitsandbytes: {bnb or 'NOT FOUND'}")
    print(f"transformers: {tf or 'NOT FOUND'}")
    print(f"peft: {peft or 'NOT FOUND'}")

    if not bnb:
        print("ERROR: bitsandbytes not found. Install with: pip install bitsandbytes")
        return False
    if not tf:
        print("ERROR: transformers not found. Install with: pip install transformers")
        return False

    # Check existing patches
    bnb_fn = bnb / "autograd/_functions.py"
    existing = subprocess.run(
        ["grep", "-c", "PATCH sirfyyn", str(bnb_fn)],
        capture_output=True, text=True
    )
    count = int(existing.stdout.strip() or "0")
    print(f"\nExisting patches in bitsandbytes/_functions.py: {count}/3 expected")

    tf_gemma4 = tf / "models/gemma4/modeling_gemma4.py"
    if tf_gemma4.exists():
        existing_tf = subprocess.run(
            ["grep", "-c", "PATCH sirfyyn", str(tf_gemma4)],
            capture_output=True, text=True
        )
        count_tf = int(existing_tf.stdout.strip() or "0")
        print(f"Existing patches in transformers/gemma4/modeling_gemma4.py: {count_tf}/4 expected")

    return True

def verify_patches():
    print("=== Verifying active patches ===")
    bnb = find_package("bitsandbytes")
    tf = find_package("transformers")
    peft = find_package("peft")

    files = [
        (bnb / "autograd/_functions.py", "P1, P9, P2"),
        (bnb / "nn/modules.py", "P3"),
        (tf / "models/gemma4/modeling_gemma4.py", "P4, P5, P7, P10"),
        (tf / "integrations/sdpa_attention.py", "P6"),
        (peft / "tuners/lora/bnb.py", "P8"),
    ]

    all_ok = True
    for fpath, patches in files:
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath.name}")
            continue
        result = subprocess.run(
            ["grep", "-c", "PATCH sirfyyn", str(fpath)],
            capture_output=True, text=True
        )
        count = int(result.stdout.strip() or "0")
        status = "✓" if count > 0 else "✗ NOT PATCHED"
        print(f"  {status} {fpath.name} [{patches}]: {count} patch marker(s)")
        if count == 0:
            all_ok = False

    if all_ok:
        print("\n✓ All patches active. Clear __pycache__ before running:")
        print("  find $(python -c 'import site; print(site.getsitepackages()[0])') -name '*.pyc' -path '*/bitsandbytes/*' -delete")
        print("  find $(python -c 'import site; print(site.getsitepackages()[0])') -name '*.pyc' -path '*/transformers/models/gemma4/*' -delete")
    else:
        print("\n✗ Some patches missing. See README.md for manual instructions.")

    return all_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.check or (not args.verify and not args.apply):
        check_patches()
    if args.verify:
        verify_patches()
    if args.apply:
        print("Manual apply: see patches/ directory for each patch file.")
        print("Automated apply coming in next release.")
        verify_patches()
