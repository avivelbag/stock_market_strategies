# Strategy Lab — the visual gallery

A zero-build, zero-dependency static site that makes every backtest in this
repo *visualisable and self-teaching*. It mirrors the structure of
`~/Documents/git/computer_art`: a gallery `index.html` that reads a
`pieces.json` manifest and renders a themed card grid, where each card opens a
self-contained piece page.

The difference from the art gallery is the source of truth: instead of bespoke
generative code per piece, a single Python build step (`build.py`) runs the
**real backtest engine** and precomputes the JSON every page renders. The
gallery therefore can never disagree with `strategies/*/metrics.json`.

## Run it

```bash
# one-time: scientific deps for the build step
python3 -m venv .venv && . .venv/bin/activate
pip install numpy pandas scipy

# 1. generate the data (run from the repo root)
python3 viz/build.py

# 2. serve the static site (fetch() needs http://, not file://)
cd viz && python3 -m http.server
# open http://localhost:8000
```

## Layout

```
viz/
  index.html        gallery: reads pieces.json, renders cards + theme toggle
  styles.css        shared theme (CSS vars drive light/dark, charts included)
  pieces.json       manifest: one card per strategy + the leaderboard
  build.py          runs the engine, writes data/*.json + thumbnails
  assets/
    chart.js        zero-dependency canvas charting (one function)
    piece.js        builds a strategy page from its JSON
    leaderboard.js  builds the comparison dashboard
  data/             generated JSON (committed, like art's piece.svg)
  pieces/
    <strategy>/index.html   thin shell: sets STRATEGY_ID, loads assets
    <strategy>/thumbnail.svg generated static snapshot
    <strategy>/README.md     what that piece teaches
    leaderboard/...          the dashboard piece
```

`data/*.json` and `pieces/*/thumbnail.svg` are generated artefacts but are
committed so the site stays a pure static site (same choice computer_art makes
with its committed `piece.svg`/`thumbnail.svg`). Regenerate them any time with
`python3 viz/build.py`.

## Educational design

Every strategy page leads with the *economic idea* and the exact *signal rule*,
then runs the unchanged rule across all four synthetic regimes via dataset
tabs. The teaching moment is the contrast — same logic, opposite outcomes — and
the explicit gap between the in-sample headline and the walk-forward truth. The
leaderboard distils that into one left-to-right colour fade.

## Swapping in real data

The pages only ever read `viz/data/*.json`; they contain no strategy maths and
no dataset assumptions. To run on real instruments:

1. Drop real OHLCV CSVs into `data/` (same columns: `date,open,high,low,close,
   volume`; a parseable date index).
2. In `build.py`, point `DATASETS` at the new file stems and update the
   `REGIME` descriptions (and, if a real series isn't long enough for the
   252-bar / walk-forward warm-ups, expect the same uninformative-OOS caveat
   the synthetic 52-week page already explains honestly).
3. Re-run `python3 viz/build.py`. No page or asset code changes.

The engine, the fill model, and the metrics are unchanged — that is the point:
the visualisation is a lens on the existing framework, not a reimplementation.
