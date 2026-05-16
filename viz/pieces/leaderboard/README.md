# The Leaderboard — visualised

A dashboard that puts all four strategies side by side, built from
`RANKING.md`'s five criteria and every strategy's `metrics.json`.

## What it teaches

One idea, hammered home: **a green in-sample Sharpe routinely turns red
out-of-sample.** The comparison table is laid out so you read left → right —
the four in-sample Sharpe columns, then the walk-forward columns — and watch
the colour drain away. That fade is the single most important lesson in
quantitative strategy evaluation: in-sample performance is a hypothesis;
walk-forward is the test.

It also makes the five ranking criteria explicit (Thesis, Robustness,
Simplicity, Walk-forward, Raw performance) and shows the full equity-curve
small-multiples grid — every strategy against every regime, against buy & hold.

## What you see

1. **Criteria** — the five dimensions strategies are judged on.
2. **In-sample vs out-of-sample table** — per-regime Sharpe heat-mapped, mean
   columns, and the two walk-forward columns that actually decide the ranking.
3. **Equity small-multiples** — 4 strategies × 4 regimes of equity (solid) vs
   buy & hold (dashed), all normalised to ×1.00.

## Files

- `index.html` — thin shell; loads the shared assets and `leaderboard.js`.
- `thumbnail.svg` — static bar snapshot of mean OOS consistency, regenerated
  by `viz/build.py`.
- `README.md` — this file.

Data is `viz/data/leaderboard.json` plus each strategy's JSON, all produced by
the real engine.
