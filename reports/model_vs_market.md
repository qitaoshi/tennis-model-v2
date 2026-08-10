# Model versus the bookmakers, as forecasters

The market's de-vigged probability scored as a rival forecast, with the same metrics as the model, on the same matches. No stake, no price, no margin — this is an accuracy comparison, not a betting one.


**Scope: TUNE only (2024-01-01 .. 2025-06-30), ATP main tour.** The odds file runs to 2026-05 and so covers TEST and HOLDOUT as well; both are already spent, so this cuts at the TUNE boundary rather than quietly reading them. Nothing is fitted here.


## Consensus of all books


| forecaster | Brier | log-loss | ECE |
|---|---:|---:|---:|
| market (median of books) | 0.18354 | 0.54380 | 0.02970 |
| model (calibrated) | 0.21522 | 0.61730 | 0.02027 |
| model (uncalibrated) | 0.21508 | 0.61706 | 0.01199 |


1,103 matches. Gap to market: Brier +0.03168, log-loss +0.07350. Positive means the model is worse.


## Per bookmaker

| bookmaker   |   n_matches |   market_brier |   market_logloss |   market_ece |   model_brier |   model_logloss |   model_ece |   model_raw_brier |   brier_gap |   logloss_gap |
|:------------|------------:|---------------:|-----------------:|-------------:|--------------:|----------------:|------------:|------------------:|------------:|--------------:|
| GGBET       |        1104 |       0.184658 |         0.545822 |    0.0272631 |      0.215138 |        0.617128 |   0.020256  |          0.215018 |   0.0304794 |     0.0713057 |
| 1xBet       |        1097 |       0.184631 |         0.546777 |    0.0197333 |      0.215905 |        0.619029 |   0.0198312 |          0.215763 |   0.0312742 |     0.0722527 |
| 22Bet       |        1097 |       0.183589 |         0.543713 |    0.0239316 |      0.215208 |        0.61716  |   0.020835  |          0.215094 |   0.0316196 |     0.0734468 |
| bet365      |        1097 |       0.182791 |         0.542352 |    0.0498705 |      0.214394 |        0.615531 |   0.0208626 |          0.214292 |   0.0316029 |     0.0731787 |
| Betsson     |        1076 |       0.183993 |         0.545242 |    0.0335924 |      0.214844 |        0.616586 |   0.0226223 |          0.214722 |   0.0308508 |     0.071344  |
| BetInAsia   |        1030 |       0.181793 |         0.539153 |    0.0288499 |      0.212657 |        0.611449 |   0.0248788 |          0.212667 |   0.030863  |     0.0722955 |
| N1 Bet      |         490 |       0.189035 |         0.556539 |    0.0365697 |      0.232262 |        0.655166 |   0.0678459 |          0.231403 |   0.0432278 |     0.0986271 |


## Where they disagree


On the 545 matches where model and market differ by more than 10 points, market Brier 0.16509 against model 0.22561. This is the honest test: agreeing with the market costs nothing and proves nothing, so the model's value shows up only where it takes a different view.


## Reading this

A bookmaker's price is close to the best public forecast that exists for a tennis match — it aggregates sharp money, injury news and team information the model has none of. So the target is not to beat it. The useful question is how much is given up, and whether that gap is small enough for the model's projections to be worth acting on where no line exists.


## Other markets

Consensus quote per selection, pushes excluded (a push is neither a win nor a loss and cannot be scored as either). Positive `brier_gap` means the model is worse.

| market         |   n_selections |   n_matches |   market_brier |   model_brier |   market_logloss |   model_logloss |   market_ece |   model_ece |   brier_gap |     corr |    mean_diff |   base_rate |
|:---------------|---------------:|------------:|---------------:|--------------:|-----------------:|----------------:|-------------:|------------:|------------:|---------:|-------------:|------------:|
| total_games    |          25760 |        1095 |       0.217006 |      0.22162  |         0.62215  |        0.633269 |    0.0230078 |   0.0199742 |  0.00461428 | 0.894223 |  0.000140543 |     0.5     |
| games_handicap |           7494 |         818 |       0.198699 |      0.217343 |         0.580383 |        0.625554 |    0.0118919 |   0.0623573 |  0.0186445  | 0.84063  | -4.10735e-15 |     0.5     |
| set_betting    |           5356 |        1099 |       0.142339 |      0.14985  |         0.446538 |        0.468561 |    0.0171187 |   0.0101126 |  0.00751162 | 0.832892 | -2.89805e-05 |     0.20519 |
| total_sets     |           3082 |        1058 |       0.211161 |      0.215097 |         0.610188 |        0.619266 |    0.0296257 |   0.017371  |  0.00393611 | 0.918222 | -5.90026e-15 |     0.5     |
| match_winner   |           2206 |        1103 |       0.183537 |      0.215221 |         0.543797 |        0.617303 |    0.0297006 |   0.0117907 |  0.0316845  | 0.785876 |  1.13583e-17 |     0.5     |

