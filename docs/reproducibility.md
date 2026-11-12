# Reproducibility guide

## Scope

The exact manuscript figures are preserved under `outputs/paper_figures/`. The scripts under `src/` reproduce the underlying analyses when the large inputs described in `data/README.md` are available. Compact result files under `results/` allow the reported numerical summaries to be checked without rerunning every experiment.

## Line 1 Temporal prioritisation

1. Copy and edit `config/line1.example.json` as `config/line1.json`.
2. Run the stages implemented in `src/line1/experimento.py`: audit, prepare, score, and evaluate.
3. Run `src/line1/comparar_blanqueamiento.py` for the four comparison geometries.
4. Run `src/line1/analisis_robusto.py` and `src/line1/sensibilidad_confianza.py` for the spatial and resampling checks.

These scripts support Tables 4 and 5 and Figures 1 and 2.

## Line 2 Institutional-record retrieval

The current manuscript uses the later recurrence experiment, not the earlier K=5 connected-component package.

1. `representation_comparison/run_iterative.py` compares the original cosine, whitened cosine, and background-centred cosine used in Figure 3.
2. `whitening_sensitivity/run_factorial.py` evaluates whitening intensity and regularisation.
3. `penalty_sensitivity/run_local.py` evaluates the external penalties shown in Figure 4.
4. `final_recurrence/run_local.py` executes twenty seeds with penalty 5 and at most 400 accepted rules.
5. `final_recurrence/actualizar_figuras.py` creates the stability and frequency products used in Figures 5 and 6.
6. `spatial_transfer/transfer.py` evaluates north-to-south and south-to-north transfer for Table 8 in the manuscript.

All experiments read the same prepared Line 2 directory. Set `SATEMB_LINE2_DATA` when it is stored outside `data/raw/line2`.

## Line 3 Conditional scenarios

1. Run `src/line3/linea3_pipeline.py` for the principal geometric experiment and its sensitivity configurations.
2. Run `src/line3/linea3_validacion_transformaciones.py` to verify norms, angles, and mean squared errors.
3. Run `src/line3/linea3_downstream_espectral.py` for the fixed NDMI and NDVI Ridge models.
4. Run `src/line3/linea3_sintesis_publicacion.py` to generate the compact publication tables and figures.
5. Run `src/line3/linea3_ejemplo_transformacion.py` only when the 64 component rasters required by Figure 8 are available.

Set `SATEMB_LINE3_DATA` when the prepared inputs are stored outside `data/raw/line3`.

## Verification

Run:

```bash
python tests/check_repository.py
python -m unittest discover -s src/line1 -p "test_*.py" -v
```

The first command checks the public repository structure and Python syntax. The Line 1 tests exercise the principal geometry and validation safeguards without requiring the complete territorial rasters.

## Archiving

Before submission or publication:

1. create a versioned GitHub release;
2. archive that release with Zenodo;
3. add the resulting DOI to `CITATION.cff`, the README, and the manuscript's code-availability statement; and
4. cite the archived software release in the manuscript references.
