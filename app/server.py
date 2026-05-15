"""Local web demo: play against the engine from a browser.

Run:
    python -m uvicorn app.server:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chess_engine import Evaluator, search_best_move

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT.parent / "data"


def _load_learned(name: str) -> Evaluator | None:
    path = DATA_DIR / f"weights_{name}.npy"
    if not path.exists():
        return None
    return Evaluator(np.load(path).astype(np.float32), name=f"learned_{name}")


def _build_evaluators() -> dict[str, Evaluator]:
    evals: dict[str, Evaluator] = {}
    for name in ("logistic", "ls", "ridge"):
        ev = _load_learned(name)
        if ev is not None:
            evals[f"learned_{name}"] = ev
    evals["psqt_handcrafted"] = Evaluator.psqt_handcrafted()
    evals["material_only"] = Evaluator.material_only()
    return evals


EVALUATORS = _build_evaluators()
DEFAULT_EVALUATOR = next(iter(EVALUATORS))


class MoveRequest(BaseModel):
    fen: str
    depth: int = Field(default=3, ge=1, le=5)
    evaluator: str = DEFAULT_EVALUATOR


class MoveResponse(BaseModel):
    move: str | None
    fen_after: str
    score: float
    nodes: int
    game_over: bool
    result: str | None


app = FastAPI(title="Chess engine demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/evaluators")
def list_evaluators() -> dict:
    return {"evaluators": list(EVALUATORS.keys()), "default": DEFAULT_EVALUATOR}


@app.post("/api/engine_move", response_model=MoveResponse)
def engine_move(req: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid FEN: {e}")

    if req.evaluator not in EVALUATORS:
        raise HTTPException(status_code=400, detail=f"unknown evaluator: {req.evaluator}")

    evaluator = EVALUATORS[req.evaluator]

    if board.is_game_over(claim_draw=True):
        return MoveResponse(
            move=None,
            fen_after=board.fen(),
            score=0.0,
            nodes=0,
            game_over=True,
            result=board.result(claim_draw=True),
        )

    result = search_best_move(board, depth=req.depth, evaluator=evaluator)
    if result.move is None:
        return MoveResponse(
            move=None,
            fen_after=board.fen(),
            score=result.score,
            nodes=result.nodes,
            game_over=True,
            result=board.result(claim_draw=True),
        )

    board.push(result.move)
    over = board.is_game_over(claim_draw=True)
    return MoveResponse(
        move=result.move.uci(),
        fen_after=board.fen(),
        score=result.score,
        nodes=result.nodes,
        game_over=over,
        result=board.result(claim_draw=True) if over else None,
    )
