# Multi-market CLV and ROI backtest

This is evaluation only; no model parameter was fitted, tuned, or selected.
ROI uses **raw Bet365 decimal odds** (`decimal_odds - 1` on a win), while
de-vigged `market_p` is used only for edge and CLV. A push returns zero profit
and remains in the placed-bet denominator.

## Known-good Bet365 reconciliation

2,115 side comparisons; agreement within 0.05 decimal odds 99.6%; median absolute difference 0.000; worst 5.670

The exact OddsPortal bookmaker tag is `bet365`. The reconciliation is a gate:
other families must not be interpreted until the scraped match-winner prices
agree closely with the independent B365W/B365L reference.

## Family results

### match_winner

The model's preferred selections had mean CLV +nan%
(bootstrap 95% CI +nan% to +nan%) versus the sharp
quote. Betting positive-Bet365-edge selections returned -5.15%
per unit staked at Bet365's raw price (bootstrap 95% CI
-11.71% to +1.68%).

- coverage: 2,201 matched, 1,097 with Bet365,
  584 bets, 1,104 Bet365 exclusions
- verdict: **no demonstrated edge**

### total_games

The model's preferred selections had mean CLV +nan%
(bootstrap 95% CI +nan% to +nan%) versus the sharp
quote. Betting positive-Bet365-edge selections returned -2.69%
per unit staked at Bet365's raw price (bootstrap 95% CI
-7.44% to +1.92%).

- coverage: 14,464 matched, 1,583 with Bet365,
  1,508 bets, 12,881 Bet365 exclusions
- verdict: **no demonstrated edge**

### total_sets
No matched odds were available.

### games_handicap

The model's preferred selections had mean CLV +nan%
(bootstrap 95% CI +nan% to +nan%) versus the sharp
quote. Betting positive-Bet365-edge selections returned -10.60%
per unit staked at Bet365's raw price (bootstrap 95% CI
-18.15% to -2.99%).

- coverage: 4,378 matched, 611 with Bet365,
  555 bets, 3,767 Bet365 exclusions
- verdict: **no demonstrated edge**

### set_betting

The model's preferred selections had mean CLV +nan%
(bootstrap 95% CI +nan% to +nan%) versus the sharp
quote. Betting positive-Bet365-edge selections returned -16.84%
per unit staked at Bet365's raw price (bootstrap 95% CI
-28.75% to -3.15%).

- coverage: 10,204 matched, 5,350 with Bet365,
  2,314 bets, 4,854 Bet365 exclusions
- verdict: **no demonstrated edge**
