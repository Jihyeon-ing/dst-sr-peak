# Code and Event Catalog for "A Data-Driven Empirical Formula of Dst Minimum Using Symbolic Regression"

Jihyeon Son, Yong-Jae Moon, Youngjae Kim, Young-Sil Kwak, and Kyung Sun Park

This archive contains the analysis code and the storm event catalog needed to reproduce
the results and figures of the paper. The discovered relation is

    Dst_peak = -3.15 * sum_{k=1..5} Ey_rect(t_p - k) - 23.6  [nT]

where Ey_rect = max(-V*Bz, 0) is the rectified dawn-to-dusk solar wind electric field
(rectified at 5-min resolution, then hourly averaged) and t_p is the storm peak time.

## Contents

```
event_catalog/
  storm_catalog.csv        354 storm events (peak_time_UT, Dst_peak_nT, subset)
                           333 training events (1995-2024), 21 test events (2025-2026)
code/
  dst_pysr_cv.py                    shared constants and helpers (SEED=42, Dst loader)
  dst_pysr_temporal_test_claude.py  train/test split, PySR model factory, gain-efficiency rule
  dst_pysr_temporal_inp2.py         OMNI preprocessing and the 891-channel feature set
  dst_intensity_multivar.py         expression parsing helper
  dst_t0_pipeline.py                main pipeline: screening -> combined search -> final fit
  dst_extlag10_t0.py                window-extension rerun (lags to t-10)
  dst_memory_kernel.py              box-window and exponential-kernel (tau) scans
  dst_coupling_benchmark.py         classical coupling-function benchmark
  dst_coupling_bootstrap.py         paired bootstrap confidence intervals (10,000 resamples)
  dst_coupling_optwindow.py         per-function optimal integration windows
  dst_offset_validity.py            validity of the relation away from the peak
  dst_exponent_test.py              free-exponent (linearity) bootstrap test
  make_paper_figures.py             paper figures
requirements.txt
```

## Data access

The raw data are not redistributed here; both sources are publicly available.

- 5-min OMNI solar wind / IMF data: NASA/GSFC OMNIWeb, https://omniweb.gsfc.nasa.gov/
  (place yearly text exports under `OMNI_5m/`; columns: year, doy, hour, minute,
  B, Bx, By, Bz, V, N, T as used in `dst_pysr_temporal_inp2.omni_preprocess_ext`)
- Hourly Dst index: World Data Center for Geomagnetism, Kyoto,
  https://wdc.kugi.kyoto-u.ac.jp/

The storm events were identified from the Dst series (Dst below -50 nT for at least 3 h,
first minimum within 24 h retained, manual inspection for a well-defined single minimum);
`event_catalog/storm_catalog.csv` lists the final 354 events so that all results can be
reproduced without repeating the manual step.

## Environment

Python 3.11, PySR 1.5.9 (which installs its Julia backend on first use). Install with

    pip install -r requirements.txt

All SR runs use `deterministic=True`, `parallelism="serial"`, and fixed random seeds
(42 + fold index for CV folds, 142 for the final fit), so results are exactly reproducible.

## Reproducing the paper

| Step | Command | Paper result |
|---|---|---|
| Main pipeline | `python dst_t0_pipeline.py` | Eq. (2), Table 1, Fig. 1 |
| Window extension | `python dst_extlag10_t0.py` | five-hour window robustness |
| Kernel scans | `python dst_memory_kernel.py` | tau = 3.5 h, Fig. 2 |
| Benchmark | `python dst_coupling_benchmark.py` then `python dst_coupling_bootstrap.py` | Table 2, bootstrap CIs |
| Window fairness | `python dst_coupling_optwindow.py` | 3-6 h optimal windows |
| Offset validity | `python dst_offset_validity.py` | Fig. 4 |
| Linearity test | `python dst_exponent_test.py` | free-exponent CIs |
| Figures | `python make_paper_figures.py` | Fig. 1-4 |

Each script writes its outputs (CSV tables, JSON summaries, and a report) to its own
output directory. A full pipeline run takes several hours on a single core because the
searches are deterministic and serial.

## License and citation

License: MIT (see LICENSE).
If you use this code or the event catalog, please cite the paper and this archive
(Zenodo DOI: [TO BE FILLED AFTER UPLOAD]).
