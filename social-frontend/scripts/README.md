# Dataset build utilities

`build_pheme_dataset.py` converts protected raw PHEME threads into the JSON
contract consumed by `social-frontend/public/datasets/`.

The script is not part of the browser runtime. It is retained because it is the
reproducible source for the checked-in PHEME demonstration datasets. It reads:

- `data/raw/pheme/`
- the v5 Text RF historical OOF ranking declared at the top of the script

and writes the event `thread.json` files plus `manifest.json` under
`social-frontend/public/datasets/`.

Run it from the repository root with:

```powershell
python .\social-frontend\scripts\build_pheme_dataset.py
```

Review the configured input paths before running. The protected raw data and
model reference artifacts are intentionally excluded from Git and must be
present on the workstation.
