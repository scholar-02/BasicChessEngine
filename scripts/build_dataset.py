"""Sample positions from a PGN, label them with Stockfish, save as parquet.

Output columns: game_id, ply, fen, result, stockfish_cp.

    python scripts/build_dataset.py --n 100      # quick smoke test
    python scripts/build_dataset.py --n 10000    # full dataset
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

RESULT_TO_INT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


# The winget Stockfish package ships an exe named after the cpu build
# (stockfish-windows-x86-64-avx2.exe), so plain which("stockfish") misses it.
_STOCKFISH_NAMES = (
    "stockfish",
    "stockfish-windows-x86-64-avx2",
    "stockfish-windows-x86-64-bmi2",
    "stockfish-windows-x86-64-sse41-popcnt",
    "stockfish-windows-x86-64",
    "stockfish-ubuntu-x86-64-avx2",
    "stockfish-macos-arm64-apple-silicon",
)


def find_stockfish(override: str | None) -> str:
    if override:
        return override
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    for name in _STOCKFISH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    sys.exit(
        "Could not locate Stockfish. Install it and either put its directory "
        "on PATH, set $env:STOCKFISH_PATH to the full exe path, or pass "
        "--stockfish <path>."
    )


def iter_games(pgn_path: Path):
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        gid = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            yield gid, game
            gid += 1


def sample_positions(
    pgn_path: Path,
    n_target: int,
    *,
    skip_opening_plies: int = 8,
    skip_ending_plies: int = 4,
    per_game_max: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    pbar = tqdm(total=n_target, desc="sampling positions")
    for gid, game in iter_games(pgn_path):
        if len(rows) >= n_target:
            break

        result = game.headers.get("Result", "*")
        if result not in RESULT_TO_INT:
            continue

        board = game.board()
        positions: list[tuple[int, str]] = []
        for ply, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            positions.append((ply, board.fen()))

        usable = [
            (ply, fen)
            for ply, fen in positions
            if skip_opening_plies <= ply <= len(positions) - skip_ending_plies
        ]
        if not usable:
            continue

        n_pick = min(per_game_max, len(usable), n_target - len(rows))
        for ply, fen in rng.sample(usable, k=n_pick):
            rows.append(
                {"game_id": gid, "ply": ply, "fen": fen, "result": RESULT_TO_INT[result]}
            )
            pbar.update(1)
    pbar.close()

    return pd.DataFrame(rows)


def label_with_stockfish(
    positions: pd.DataFrame,
    stockfish_path: str,
    depth: int,
    mate_score: int = 10_000,
) -> pd.DataFrame:
    scores: list[int] = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for fen in tqdm(positions["fen"], desc="labeling with stockfish"):
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            cp = info["score"].white().score(mate_score=mate_score)
            scores.append(int(cp))
    out = positions.copy()
    out["stockfish_cp"] = scores
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pgn", type=Path, default=DATA_DIR / "lichess_sample.pgn",
                   help="input PGN file")
    p.add_argument("--n", type=int, default=10_000,
                   help="target number of positions")
    p.add_argument("--depth", type=int, default=12,
                   help="Stockfish search depth per position")
    p.add_argument("--stockfish", type=str, default=None,
                   help="path to Stockfish binary (else $STOCKFISH_PATH or PATH)")
    p.add_argument("--out", type=Path, default=DATA_DIR / "labeled_positions.parquet",
                   help="output parquet file")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pgn.exists():
        sys.exit(
            f"PGN file not found: {args.pgn}\n"
            "Download a sample from https://database.lichess.org/ and place it here."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    stockfish = find_stockfish(args.stockfish)
    print(f"using stockfish: {stockfish}")

    positions = sample_positions(args.pgn, args.n, seed=args.seed)
    print(f"sampled {len(positions)} positions from {positions['game_id'].nunique()} games")

    labeled = label_with_stockfish(positions, stockfish, args.depth)
    labeled.to_parquet(args.out, index=False)
    print(f"wrote {len(labeled)} rows to {args.out}")


if __name__ == "__main__":
    main()
