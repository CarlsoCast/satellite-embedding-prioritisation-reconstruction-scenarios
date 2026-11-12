# Data requirements

Raw data are not committed to this repository because several inputs are large and retain their original access and licensing conditions. Keep the directory structure below or set the environment variables documented in `config/paths.example.env`.

## Line 1

Place the 2022 and 2025 AlphaEarth Foundations rasters and the corresponding annual Dynamic World probability composites under `data/raw/line1/`. Update `config/line1.json` with the actual filenames. Dynamic World inputs must preserve the nine-class band order declared in the configuration.

## Line 2

The prototype-retrieval experiments expect a prepared Line 2 directory containing:

```text
data/raw/line2/
├── comparacion_alineada/X_unit.dat
└── nueva_corrida/resultados_linea2_validacion_espacial_folds/_cache/
    ├── config.json
    ├── row.dat
    ├── col.dat
    └── risk.dat
```

`X_unit.dat` contains normalised 64-component embeddings. The cache files contain the common grid and the rasterised institutional flood-hazard reference. An external directory with this structure can be supplied through `SATEMB_LINE2_DATA`.

## Line 3

Place the prepared window-level inputs under `data/raw/line3/`:

```text
data/raw/line3/
├── datos_para_procesar/alphaearth_dynamicworld_residual/
│   └── ventanas_dynamicworld_alphaearth_residual.parquet
├── resultados_60_downstream_unificado_lst_bsa/
│   ├── 01_objetivos_lst_bsa_anuales.parquet
│   ├── 02_controles_hidrologicos.parquet
│   └── 03_base_autocontenida.parquet
├── experimento_ndvi_ndmi_20261014/
├── experimento_cuatro_variables_familias_20261014/
└── DATOS_ALINEADOS/
```

An external Line 3 directory can be supplied through `SATEMB_LINE3_DATA`. The optional `SATEMB_RASTER_DIR` variable points to the 64 component rasters used only for the detailed example in Figure 8.

## Sources and terms

- AlphaEarth Foundations and Dynamic World were accessed through Google Earth Engine. Users must obtain their own access and comply with the applicable Google Earth Engine and product terms.
- NDVI and NDMI references were derived from Sentinel-2 MSI Level-2A imagery.
- Terrain variables were derived from the Bogotá digital terrain model distributed through IDECA.
- The institutional flood-hazard reference corresponds to the mapping used in Bogotá's 2021 Land Use Plan.

The MIT licence in this repository applies to the original code, not to these source datasets.
