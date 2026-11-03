# PsiSWRs
Code to analyze data and generate figures for [Bozkir et al. (2026), *bioRxiv*](https://www.biorxiv.org/content/10.64898/2026.08.17.745042v1) which determines the effect of psilocybin (10 mg/kg) on hippocampal sharp-wave ripples (SWRs) in rats.

## Requirements
1) Python
2) This repository
3) Diba lab NeuoroPy repository: https://github.com/diba-lab/NeuroPy
4) Data (to be provided upon publication)

## Organization
### SWR feature extraction
- `swr_features_during_wake_states.py`: extracts SWR features (duration, peak frequency, peak power, sharp-wave amplitude) for WAKE states only.
- `swr_features_across_entire_session.py`: extracts SWR features including across all states.

Default is to only include 1st hour after injection to match recording duration across all sessions (SALINE1 - Day 1, PSILOCYIN - Day 2, SALINE2 - Day 3). Options to include full session length in each function

### Sleep scoring
- Sleep scoring was performed using [SleepScoreMaster](https://github.com/buzsakilab/buzcode/tree/master/detectors/detectStates/SleepScoreMaster) code.
- Python scripts in the `sleepscoring` folder generate hypnograms, scatterplots of brainstate metrics, and quantification of non-rapid eyemovement (NREM) sleep and high voltage spindles (HVS)
