"""Bounded materialization smoke: ten threads per event, all checkpoints."""
from materialize_multicheckpoint_dataset import run


if __name__ == "__main__":
    run(mode="smoke", max_threads_per_event=10)
