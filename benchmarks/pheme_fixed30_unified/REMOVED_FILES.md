# Removed files

Legacy materialization and four-model training runners were replaced by `run.py`,
which validates the immutable contract and evaluates frozen OOF predictions.
Models are trained only in their own canonical folders. Historical code remains
available in Git history and the authoritative full output remains in `reference_result/`.
