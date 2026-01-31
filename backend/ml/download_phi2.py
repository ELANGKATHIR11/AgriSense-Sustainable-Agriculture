"""Phi-2 downloader removed.

GPU/NPU-related download/quantization utilities were removed to keep the
project CPU-only. This file is retained as a safe stub for historical
reference.

The file intentionally contains no runtime logic other than a short
informational message.
"""


def main() -> None:
    """Inform about removed GPU/NPU features."""
    print(
        "[INFO] Phi-2 download/quantization script removed (GPU/NPU features stripped)."
    )


if __name__ == "__main__":
    main()
