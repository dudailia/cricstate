| model | test NLL [95% CI] | note |
|---|---|---|
| M_state | 1.61438 [1.60858, 1.61995] |  |
| M_shrunk | 1.60934 [1.60360, 1.61489] | empirical-Bayes shrunk striker+bowler effects |
| M_flat | 1.76775 [1.75993, 1.77592] | unshrunk MLE — overfits sparse players |
| M_shuffled | 1.61533 [1.60952, 1.62090] | shuffled-identity leakage canary |

Primary ΔNLL (shrunk minus state): **-0.00504 [-0.00561, -0.00449]** = 0.31% relative, 0.0073 bits/ball. Verdict: **AMBIGUOUS**.

Dilution: null-striker 5.25%, unseen striker 14.45%, unseen bowler 19.15%.
