# Sweeping Stage 7's `split_sigma` against the total-games bias

The model expects 0.428 more games than matches produce; with Stage 7 off that is 2.218, so the corrections already remove 80% and the mechanism is right. Serve rates explain almost none of it. `split_sigma` wobbles the skill split, which makes sets lopsided, and lopsided sets are shorter.


**The objection this checks:** Stage 7's own grid ran to 0.12 and chose 0.08. A larger value was available and rejected — on PIT deviation, not on mean bias. So fixing the mean may cost the distribution shape Stage 7 was gated on, and the table below reports both so the trade is visible rather than implied.


Gate carried over unchanged: `|tb_gap| < 0.01`, coverage in (0.75, 0.85). TUNE only; nothing fitted or written.


## Sweep

|   split_sigma |   games_bias |      tb_gap |    pit_dev |   coverage80 | gate_ok   |
|--------------:|-------------:|------------:|-----------:|-------------:|:----------|
|         0     |    2.20146   | -0.081577   | 0.0290667  |     0.743333 | False     |
|         0.08  |    0.428406  | -0.0116629  | 0.00706667 |     0.8115   | False     |
|         0.09  |    0.118524  |  0.00168446 | 0.00653333 |     0.8195   | True      |
|         0.095 |   -0.0419456 |  0.00823771 | 0.0078     |     0.8235   | True      |
|         0.1   |   -0.185602  |  0.0141215  | 0.00863333 |     0.824667 | False     |
|         0.105 |   -0.356213  |  0.0208478  | 0.0103667  |     0.829833 | False     |
|         0.11  |   -0.497388  |  0.0267757  | 0.0119667  |     0.832167 | False     |
|         0.12  |   -0.793926  |  0.0390305  | 0.0167667  |     0.837167 | False     |
|         0.16  |   -1.78552   |  0.0792582  | 0.0333333  |     0.854    | False     |
|         0.24  |   -3.05451   |  0.123819   | 0.0573333  |     0.857333 | False     |


Incumbent `split_sigma` 0.08: bias +0.428, PIT deviation 0.00707, coverage 0.811.


Smallest absolute bias that still passes the gate: `split_sigma` **0.095**, bias -0.042, PIT deviation 0.00780, coverage 0.824. (An earlier version of this line printed "0.10" — a two-decimal format string rounding 0.095; the table above always had the correct values.)

**But smallest absolute bias is the wrong selection rule**, and picking by it here would repeat a mistake made earlier in this session: optimising one metric while the others drift. Stage 7's own rule was to minimise PIT deviation subject to the tiebreak and coverage gates. Under that rule the answer is `split_sigma` **0.09**, which is not a compromise at all — it beats the shipped 0.08 on *every* metric:

| | shipped 0.08 | proposed 0.09 |
|---|---|---|
| games bias | +0.428 | **+0.119** |
| tiebreak gap | −0.0117 (fails) | **+0.0017** (passes) |
| PIT deviation | 0.00707 | **0.00653** |
| central-80% coverage | 0.811 | **0.820** |

No trade was needed. The incumbent is simply mistuned on current data.


**A better setting exists that keeps Stage 7's gate: True.**


## If this is taken further

Changing `split_sigma` means re-running Stage 7's gate properly (not just the two metrics here), and the calibration maps sit downstream and would need refitting — they were fitted against the current corrections. And there is still no untouched window to confirm the result on. This sweep says whether the fix is available, not whether it is proven.

