# R-SCR+

Code, per-window results 

**Counterfactual Repairs for Multivariate Time Series Anomalies That Survive Detector Retraining**
Thu Le, Nhan Vuong, Quoc Di Nguyen, Hoa Nguyen, Bac Le. ACIIDS 2027 (Springer LNAI).

R-SCR+ explains an alarm of a multivariate time series anomaly detector by restoring channels to normal in order of their deviation, stopping below the threshold with a safety margin, and accepting a repair only after a curtailed sequential vote of detector copies trained on contiguous time blocks supports it under a Beta credible bound. Every repair is scored after seed-only, full-history, later-period and chained retraining and on three detector families that took no part in the search.

## Repository layout

| Folder | Content |
|---|---|
| `code/` | Detectors, block copies, R-SCR+ and all baselines (SCR, m-SCR, p-SCR, Native-Guide and CoMTE variants, Wachter, BetaRCE, RobX, T-Rex, ROAR), retraining protocols, Wilcoxon tests with Holm correction, figure scripts |
| `out/` | Per-window results as JSON for SMD, ASD and PSM, run logs and the summary files behind every table and figure |
| 

## Data

All three benchmarks are public and are not redistributed here.

| Dataset | Source | Entities | Channels |
|---|---|---|---|
| SMD | Su et al., KDD 2019, https://github.com/NetManAIOps/OmniAnomaly | 28 machines | 38 |
| ASD | Li et al., KDD 2021, https://github.com/zhhlee/InterFusion | 12 nodes | 19 |
| PSM | Abdulaal et al., KDD 2021, https://github.com/eBay/RANSynCoders | 1 series, 16 test blocks | 25 |

Place the raw files under `data/` and run `code/prep_asd.py` for ASD and `code/export_processed.py` for the common format.

## Reproducing the results

```bash
pip install numpy scipy scikit-learn pyod pandas matplotlib
cd code
python3 prep_asd.py
bash run_journal.sh                      # all experiments, about one day on two CPU cores
bash run_v6_all.sh && python3 det_quality.py
python3 mech.py && python3 fig_mech.py
python3 analyse_j.py && python3 analyse_v5.py && python3 analyse_v6.py && python3 analyse_rev.py

```



## Citation

```bibtex
@inproceedings{le2027rscrplus,
  author    = {Le, Thu and Vuong, Nhan and Nguyen, Quoc Di and Nguyen, Hoa and Le, Bac},
  title     = {Counterfactual Repairs for Multivariate Time Series Anomalies That Survive Detector Retraining},
  booktitle = {Intelligent Information and Database Systems, ACIIDS 2027},
  series    = {Lecture Notes in Artificial Intelligence},
  publisher = {Springer},
  year      = {2027}
}
```

## Funding

This research is funded by the University of Science, VNU-HCM.

## License

MIT
