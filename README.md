# Chess engine - scientific Python semester project

A small chess engine built around `python-chess`, used as a vehicle to
demonstrate two pieces of mathematics from the scientific-Python toolkit:

- **Linear algebra** (`numpy`): bitboards as elements of `GF(2)^64`,
  position feature vectors, piece-square tables as matrices, PCA.
- **Optimization** (`scipy.optimize`, `scipy.linalg`): learning the
  evaluation function's weights from a labeled dataset of positions
  (least squares against Stockfish, ridge regression, logistic
  regression on game outcomes - i.e. Texel tuning).

The engine is intentionally small. Move generation, legality and PGN
parsing are delegated to `python-chess`. We focus on the parts where the
math lives: the evaluation function and the search.

## Layout

```
chess_engine/      reusable engine module
  psqt.py          PeSTO piece-square tables (numpy 8x8 matrices)
  features.py      phi: position -> numpy feature vector
  evaluate.py      Evaluator(weights) - score(board) = w . phi(board)
  search.py        negamax with alpha-beta pruning
  selfplay.py      play_game(white_eval, black_eval) -> chess.pgn.Game

scripts/
  build_dataset.py PGN sampling + Stockfish labeling -> parquet

notebooks/         the actual demonstration / report
  01_board_and_bitboards.ipynb
  02_features_and_psqt.ipynb
  03_search_and_pruning.ipynb
  04_learning_the_evaluation.ipynb
  05_putting_it_together.ipynb

tests/             sanity tests (pytest)
data/              PGN, generated positions and labels (gitignored)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You also need a Stockfish binary on PATH (or set `STOCKFISH_PATH`) for
the dataset-labeling script and for benchmarking. Download from
<https://stockfishchess.org/download/>.

## Quick start

```powershell
pytest -q                              # sanity tests
python scripts\build_dataset.py --n 100 # tiny end-to-end pipeline check
jupyter lab                             # open the notebooks
```

## Web demo

A small FastAPI + chessboard.js page lets you play the engine in a browser:

```powershell
python -m uvicorn app.server:app --reload
```

Then open <http://127.0.0.1:8000>. The UI exposes a search-depth slider
(1-5) and an evaluator picker (learned weights if `data/weights_*.npy`
are present, else the handcrafted PSQT / material-only baselines).
