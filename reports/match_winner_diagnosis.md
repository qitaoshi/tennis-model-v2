# Why match_winner calibrates badly on holdout

Diagnostic only — nothing is fitted or selected here, and neither TEST nor HOLDOUT is touched. Scored at the Elo layer on all pre-holdout data.


## The pattern that prompted it

Full-pipeline match_winner ECE:

| window | ECE |
|---|---|
| TUNE 2024-01..2025-06 | 0.0178 |
| TEST 2025-07..2025-12 | 0.0181 |
| HOLDOUT 2026-01..2026-07 | 0.0665 |
| (original build) TEST 2023-07..2023-12 | 0.0169 |
| (original build) HOLDOUT 2024-01..2026-07 | 0.0693 |


Both test windows are July-December; both holdout windows open in January. Drift would degrade with distance from the fitting data, and it does not — the 2025-07 window sits between the two fitting windows and calibrates fine. That points at the calendar, not at staleness.


## ECE by calendar month

|   month |     n |        ece |    brier |   mean_p |
|--------:|------:|-----------:|---------:|---------:|
|       1 |  9904 | 0.018828   | 0.220376 | 0.55747  |
|       2 | 10651 | 0.00712537 | 0.223204 | 0.554555 |
|       3 |  8857 | 0.011926   | 0.225429 | 0.551048 |
|       4 | 11555 | 0.00401351 | 0.226401 | 0.547606 |
|       5 |  9902 | 0.00867545 | 0.214771 | 0.566738 |
|       6 |  9990 | 0.012394   | 0.221265 | 0.557241 |
|       7 | 12711 | 0.00596786 | 0.225731 | 0.548942 |
|       8 | 10258 | 0.0111683  | 0.219804 | 0.561487 |
|       9 | 10545 | 0.00976109 | 0.22053  | 0.555052 |
|      10 | 11952 | 0.0123514  | 0.221576 | 0.558573 |
|      11 |  6735 | 0.0142724  | 0.223734 | 0.549247 |
|      12 |   697 | 0.0197256  | 0.227487 | 0.547671 |


## ECE by half-year

| half    |     n |        ece |    brier |   mean_p |
|:--------|------:|-----------:|---------:|---------:|
| Jan-Jun | 60859 | 0.00651197 | 0.221984 | 0.555623 |
| Jul-Dec | 52898 | 0.00428934 | 0.222375 | 0.554791 |


## ECE by layoff of the staler-rated player

Debutants are EXCLUDED here, so this isolates one thing: an existing rating that nothing has aged. A player with no prior match is a different defect with a different fix (a starting prior, not regression), and pooling the two makes one look like the other.

| layoff_days   |     n |        ece |    brier |   mean_p |
|:--------------|------:|-----------:|---------:|---------:|
| 0-7           | 38871 | 0.00548494 | 0.219997 | 0.558239 |
| 7-21          | 38412 | 0.0127018  | 0.225153 | 0.553729 |
| 21-42         | 16731 | 0.00549845 | 0.223679 | 0.554968 |
| 42-70         |  6476 | 0.00762698 | 0.220759 | 0.555787 |
| 70-140        |  5043 | 0.0314461  | 0.218793 | 0.555583 |
| 140+          |  5276 | 0.0550156  | 0.211815 | 0.557986 |


## ECE by debut status

| group              |      n |        ece |    brier |   mean_p |
|:-------------------|-------:|-----------:|---------:|---------:|
| has a debutant     |   2948 | 0.0799785  | 0.230628 | 0.530069 |
| both players known | 110809 | 0.00251501 | 0.221941 | 0.555906 |


