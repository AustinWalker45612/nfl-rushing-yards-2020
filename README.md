# Predicting NFL Rushing Yards from Player Tracking Data

Practice project on the closed 2020 NFL Big Data Bowl (Kaggle), using single-frame
player tracking data at the moment of handoff to predict rushing yards.

## The question I actually needed to answer first

What does "predicting well" mean here? Early hand-built models scored ~3.7 MAE and
I couldn't tell if that was good. The fix was switching to the competition's own
metric (CRPS — a score over the full probability distribution of yardage) and
benchmarking against a floor: a model that ignores the players entirely and just
guesses the historical yardage distribution every time. That floor scores **0.01427**.
Anything that can't beat that isn't using the tracking data for anything.

## Notebooks

- [`exploration-and-features-final.ipynb`](./exploration-and-features-final.ipynb) —
  data standardization, hand-built feature engineering, and validation of several
  defender-geometry features (closing speed, interception-margin modeling,
  engaged-vs-free defender tiers).
- [`zoo-lite-network-v1-pinned-folds.ipynb`](./zoo-lite-network-v1-pinned-folds.ipynb) —
  a small version of the competition's 1st-place network architecture.

## Results

| Approach | CRPS skill vs. floor | R² on raw yards |
|---|---|---|
| Floor (no features) | 0% | 0 |
| Hand-built features (best) | +2.4% | ~0.03 |
| Zoo-lite network (this project) | **+12.85%** | **~0.13** |
| The Zoo, 1st place (tuned + bagged) | ~+15% | — |

Hand-built distance and closing-speed features were tested one at a time under
cross-validation. Most were null. One survived — a feature summing the closing
speed of defenders within ~3 yards of an offensive lineman — giving a modest,
consistently positive gain across game-grouped and season-holdout splits.

Along the way, I found that `PlayerAccel` looked like the dominant feature
(permutation importance 2x the next-highest), but the effect was mostly a data
artifact: 2017 rows have a much stronger, physically implausible correlation
between acceleration and yards than 2018-19, consistent with known timing/
measurement issues in that season. Scoring only on 2018+ rows dropped the "real"
baseline R² from ~0.05 to ~0.024.

## A small version of the winning approach

The 2020 competition's 1st-place solution
(["The Zoo," Singer & Gordeev](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/writeups/the-zoo-1st-place-solution-the-zoo))
used a CNN over defender-vs-blocker relative position/velocity pairs, with no
hand-engineered features. I built a small version of that architecture — credited
above; this is my own implementation from their published write-up, not their code —
using the same 5 relative vectors, pooling over players, a CRPS loss, and left-right
mirror augmentation. One untuned run, 15 epochs, no bagging, got to **+12.85%** CRPS
skill over the floor — about 5x the hand-built model, and roughly 85% of the way to
the published winning score with none of their tuning or ensembling.

## Takeaway

On this dataset, the useful signal lives in *relative* player-pair geometry, not
in aggregate handoff-frame stats. That's consistent with why most of my hand-built
features — which summarized distances and speeds into single numbers per play —
came back null: they were throwing away exactly the pairwise structure the network
exploits.
