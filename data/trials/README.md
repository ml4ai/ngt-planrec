# Trials Directory

Place raw ASIST trial metadata files here. Filenames typically look like:

```
HSRData_TrialMessages_Trial-T0006xx_Team-TMxxxxxx_Member-na_CondBtwn-..._Vers-<n>.metadata
```

Notes:
- These are newline-delimited JSON records, often large (70–100 MB per file).
- The extractor scans this directory using the default glob:
  ```
  data/trials/HSRData_TrialMessages_Trial-*.metadata
  ```
- Only trials with missions `Saturn_A` or `Saturn_B` are processed.
- Lines with `mission_timer == "Mission Timer not initialized."` are ignored.

Running the extractor (example):
```bash
python utils/new_study_action_extractor.py \
  --regions data/map_excel/Saturn_2.6_3D_sm_v1.0.json \
  --output  data/new_study_actions
```


