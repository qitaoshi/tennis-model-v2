# Calibration refit — method and fitting window

Selected on TUNE (2024-01-01 .. 2025-06-30) after the 2026-08-08 re-split. **TEST and HOLDOUT were not consulted.** Maps are fitted on FIT and judged on TUNE; the shipped map is then refit on FIT+TUNE using the selected setting.


## Why this was run

The shipped maps were fitted on predictions through 2022-06 and never refreshed. match_winner ECE was 0.017 on the original TEST window and 0.069 on the original holdout. Separately, isotonic put only seven knots below p=0.20, so the tail — where the model is known to overrate its selections — was a step function fitted on very little data.


Uncalibrated baseline on TUNE: ECE 0.0188, log-loss 0.5633, Brier 0.1919.


## Grid

Ranked by ECE, then log-loss. The rule was fixed before the numbers were seen: these maps exist to fix calibration, so ECE decides, and log-loss guards against an ECE win bought by destroying sharpness.

| method   | window_years   |   n_train_matches |   tune_ece |   tune_logloss |   tune_brier |    mw_ece |   mw_logloss |
|:---------|:---------------|------------------:|-----------:|---------------:|-------------:|----------:|-------------:|
| blended  | 8              |             20000 |  0.0108233 |       0.562794 |     0.19173  | 0.0177889 |     0.663467 |
| isotonic | 8              |             20000 |  0.0114048 |       0.562983 |     0.191762 | 0.0177552 |     0.663839 |
| platt    | 8              |             20000 |  0.011426  |       0.562681 |     0.191684 | 0.0206146 |     0.663424 |
| platt    | 5              |             20000 |  0.0116324 |       0.562662 |     0.191672 | 0.0187249 |     0.663218 |
| blended  | all            |             20000 |  0.0118655 |       0.56297  |     0.19181  | 0.0218449 |     0.664084 |
| platt    | 3              |             20000 |  0.0121503 |       0.562725 |     0.191696 | 0.0209621 |     0.663464 |
| isotonic | 3              |             20000 |  0.0126658 |       0.563046 |     0.191822 | 0.023453  |     0.66457  |
| blended  | 3              |             20000 |  0.0127167 |       0.562988 |     0.191814 | 0.023661  |     0.664515 |
| isotonic | all            |             20000 |  0.0127561 |       0.563064 |     0.191827 | 0.0231008 |     0.664195 |
| platt    | all            |             20000 |  0.0130361 |       0.56289  |     0.19177  | 0.0257769 |     0.664126 |
| blended  | 5              |             20000 |  0.0135421 |       0.562921 |     0.191775 | 0.0237384 |     0.664027 |
| isotonic | 5              |             20000 |  0.0138575 |       0.563014 |     0.191792 | 0.0244783 |     0.664219 |


**Selected: `blended`, window `8`.** ECE gain over the shipped incumbent (isotonic, all history): +0.0019.


## What the selected map does to a longshot

The column that motivated the work. Model probability in, calibrated probability out.

|   model_p |   calibrated |
|----------:|-------------:|
|      0.02 |       0.0177 |
|      0.05 |       0.0457 |
|      0.08 |       0.0744 |
|      0.12 |       0.1111 |
|      0.16 |       0.1717 |
|      0.2  |       0.1844 |
|      0.3  |       0.3034 |
|      0.5  |       0.5    |


## How much of this grid is signal

Read the winner as the best of several near-ties, not as a decisive result. The ECE spread across the whole grid (0.0108 to 0.0139) is small relative to what 12,766 TUNE matches can resolve, and the window ranking is not monotone — 8 years beats both all-history and 3 years, which is not the shape a strong drift effect would make.

What the grid does support, in decreasing order of confidence: (1) recalibrating on recent data beats the stale shipped map (0.0188 uncalibrated, 0.0128 incumbent, 0.0108 selected); (2) restricting the fitting window helps; (3) which method wins matters less than which window does — at window 8 the three methods sit at 0.0108, 0.0114, 0.0114. Treat the specific method/window pair as the best available choice rather than as an established fact.

Log-loss is flat across the entire grid (0.5627 to 0.5631 against a 0.5633 baseline). That is the correct shape for a calibration map: it fixes calibration without buying it from sharpness.

## Caveat

TUNE here is 2024-01-01 .. 2025-06-30, a window the original backtest read once at aggregate level. No parameter was fitted against it then, but it is not a pristine selection set — see `constants.PRIOR_EVALUATION_WINDOWS`.

