# Calibration refit — method and fitting window

Selected on TUNE (2024-01-01 .. 2025-06-30) after the 2026-08-08 re-split. **TEST and HOLDOUT were not consulted.** Maps are fitted on FIT and judged on TUNE; the shipped map is then refit on FIT+TUNE using the selected setting.


## Why this was run

The shipped maps were fitted on predictions through 2022-06 and never refreshed. match_winner ECE was 0.017 on the original TEST window and 0.069 on the original holdout. Separately, isotonic put only seven knots below p=0.20, so the tail — where the model is known to overrate its selections — was a step function fitted on very little data.


Uncalibrated baseline on TUNE: ECE 0.0175, log-loss 0.5370, Brier 0.1823.


## Grid

Ranked by ECE, then log-loss. The rule was fixed before the numbers were seen: these maps exist to fix calibration, so ECE decides, and log-loss guards against an ECE win bought by destroying sharpness.

| method   | window_years   |   n_train_matches |   tune_ece |   tune_logloss |   tune_brier |    mw_ece |   mw_logloss | degenerate   |
|:---------|:---------------|------------------:|-----------:|---------------:|-------------:|----------:|-------------:|:-------------|
| platt    | 5              |             20000 |  0.0147904 |       0.536104 |     0.182239 | 0.0254841 |     0.65526  | False        |
| platt    | 3              |             20000 |  0.0153501 |       0.536223 |     0.182282 | 0.0282665 |     0.655832 | False        |
| platt    | 8              |             20000 |  0.0154853 |       0.536194 |     0.182273 | 0.0292514 |     0.656052 | False        |
| platt    | all            |             20000 |  0.0175243 |       0.536529 |     0.182399 | 0.0340858 |     0.657269 | False        |


**Selected: `platt`, window `5`.** ECE gain over the shipped incumbent (isotonic, all history): +0.0008.


## What the selected map does to a longshot

The column that motivated the work. Model probability in, calibrated probability out.

|   model_p |   calibrated |
|----------:|-------------:|
|      0.02 |       0.019  |
|      0.05 |       0.0482 |
|      0.08 |       0.0777 |
|      0.12 |       0.1173 |
|      0.16 |       0.1571 |
|      0.2  |       0.1971 |
|      0.3  |       0.2977 |
|      0.5  |       0.5    |


## How much of this grid is signal

Read the winner as the best of several near-ties, not as a decisive result. The ECE spread across the whole grid is small relative to what 12,766 TUNE matches can resolve, and the window ranking is not monotone — 8 years beats both all-history and 3 years, which is not the shape a strong drift effect would make.


What the grid does support, in decreasing order of confidence: (1) recalibrating on recent data beats the stale shipped map; (2) restricting the fitting window helps; (3) which method wins matters less than which window does. Treat the specific method/window pair as the best available choice rather than as an established fact.


## Caveat

TUNE here is 2024-01-01 .. 2025-06-30, a window the original backtest read once at aggregate level. No parameter was fitted against it then, but it is not a pristine selection set — see `constants.PRIOR_EVALUATION_WINDOWS`.

