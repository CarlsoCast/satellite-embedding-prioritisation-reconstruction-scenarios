# Temporal Prioritisation Signal Reconstruction and Scenario Generation Using Satellite Embeddings

This repository contains the analysis code and compact reproducibility materials for a study of three vector-space operations on annual AlphaEarth Foundations embeddings in Bogotá, Colombia:

1. temporal prioritisation through direct comparison, Procrustes alignment, and whitening;
2. recovery of patterns from an incomplete institutional flood-hazard record through prototype-based similarity retrieval; and
3. construction and evaluation of conditional embedding scenarios from observed land-cover transitions.

The repository preserves the scripts used for the reported analyses, the exact figures embedded in the submitted manuscript, and compact result tables that can be inspected without rerunning the computationally intensive stages. Raw satellite embeddings and other large source rasters are not distributed through GitHub.

## Repository structure

```text
config/                 Example path and analysis configuration
data/                   Local data location; raw data are ignored by Git
docs/                   Reproducibility and manuscript crosswalk
outputs/paper_figures/  Exact figures used in the manuscript
results/                Compact tables, audits, and derived outputs
src/line1/              Temporal prioritisation
src/line2/              Institutional-record retrieval
src/line3/              Conditional scenarios and downstream evaluation
tests/                   Repository-level checks
```

## Installation

The analyses were developed with Python 3.10. Create the environment with either Conda or pip:

```bash
conda env create -f environment.yml
conda activate satellite-embeddings
```

or

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data placement

Place local source data under `data/raw/`, following [data/README.md](data/README.md). Large files are excluded by `.gitignore`. The Line 2 and Line 3 scripts also accept external data directories through the environment variables `SATEMB_LINE2_DATA`, `SATEMB_LINE3_DATA`, and `SATEMB_RASTER_DIR`.

For Line 1, copy `config/line1.example.json` to `config/line1.json` and replace the example paths with the local AlphaEarth and Dynamic World files.

## Reproducing the analyses

The complete order of execution and the relationship between scripts and manuscript outputs are described in [docs/reproducibility.md](docs/reproducibility.md). Several full runs require tens of gigabytes of embeddings and may take hours. The compact files under `results/` reproduce the reported summaries without placing those large inputs under version control.

## Interpretation

The products in this repository have task-specific scopes:

- temporal discrepancies are priorities for review, not confirmed land-cover changes;
- external activations in Line 2 indicate similarity to recorded examples, not confirmed flood-hazard boundaries; and
- Line 3 scenarios are retrospective conditional representations, not forecasts of future territorial conditions.

## Citation

Citation metadata are provided in `CITATION.cff`. A version-specific DOI should be added after archiving a release in Zenodo.

## License

The original code in this repository is released under the MIT License. Source datasets and third-party products remain subject to their respective terms and licences.
