"""Bounded smoke for the strict-30 AI blind dataset materializer."""
from build_ai_blind_strict30_dataset import build


if __name__ == "__main__":
    build(mode="smoke", max_threads=10)
