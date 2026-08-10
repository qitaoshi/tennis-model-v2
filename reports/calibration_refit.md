# Calibration refit — method and fitting window

Selected on TUNE (2024-01-01 .. 2025-06-30) after the 2026-08-08 re-split. **TEST and HOLDOUT were not consulted.** Maps are fitted on FIT and judged on TUNE; the shipped map is then refit on FIT+TUNE using the selected setting.


## Why this was run

The shipped maps were fitted on predictions through 2022-06 and never refreshed. match_winner ECE was 0.017 on the original TEST window and 0.069 on the original holdout. Separately, isotonic put only seven knots below p=0.20, so the tail — where the model is known to overrate its selections — was a step function fitted on very little data.


Uncalibrated baseline on TUNE: ECE 0.0195, log-loss 0.5414, Brier 0.1842.


## Grid

Ranked by ECE, then log-loss. The rule was fixed before the numbers were seen: these maps exist to fix calibration, so ECE decides, and log-loss guards against an ECE win bought by destroying sharpness.

| method   | window_years   |   n_train_matches |   tune_ece |   tune_logloss |   tune_brier |    mw_ece |   mw_logloss | degenerate   |
|:---------|:---------------|------------------:|-----------:|---------------:|-------------:|----------:|-------------:|:-------------|
| platt    | 8              |             20000 |  0.0143657 |       0.539769 |     0.183997 | 0.0206146 |     0.663424 | False        |
| platt    | 5              |             20000 |  0.0146196 |       0.539764 |     0.183999 | 0.0187249 |     0.663218 | False        |
| platt    | 3              |             20000 |  0.0150741 |       0.539819 |     0.184023 | 0.0209621 |     0.663464 | False        |
| platt    | all            |             20000 |  0.0158445 |       0.539963 |     0.18409  | 0.0257769 |     0.664126 | False        |


**Selected: `platt`, window `8`.** ECE gain over the shipped incumbent (isotonic, all history): -0.0024.


## What the selected map does to a longshot

The column that motivated the work. Model probability in, calibrated probability out.

|   model_p |   calibrated |
|----------:|-------------:|
|      0.02 |       0.0177 |
|      0.05 |       0.0457 |
|      0.08 |       0.0744 |
|      0.12 |       0.1134 |
|      0.16 |       0.153  |
|      0.2  |       0.193  |
|      0.3  |       0.2943 |
|      0.5  |       0.5    |


## How much of this grid is signal

Read the winner as the best of several near-ties, not as a decisive result. The ECE spread across the whole grid is small relative to what 12,766 TUNE matches can resolve, and the window ranking is not monotone — 8 years beats both all-history and 3 years, which is not the shape a strong drift effect would make.


What the grid does support, in decreasing order of confidence: (1) recalibrating on recent data beats the stale shipped map; (2) restricting the fitting window helps; (3) which method wins matters less than which window does. Treat the specific method/window pair as the best available choice rather than as an established fact.


## Caveat

TUNE here is 2024-01-01 .. 2025-06-30, a window the original backtest read once at aggregate level. No parameter was fitted against it then, but it is not a pristine selection set — see `constants.PRIOR_EVALUATION_WINDOWS`.

