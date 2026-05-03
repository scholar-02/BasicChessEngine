"""Linear evaluator: score(board) = w . phi(board), white POV centipawns."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chess_engine.features import (
    BISHOP_PAIR_INDEX,
    MATERIAL_SLICE,
    N_FEATURES,
    PSQT_SLICES,
    SIDE_TO_MOVE_INDEX,
    phi,
)
from chess_engine.psqt import MATERIAL, PSQT


@dataclass
class Evaluator:
    weights: np.ndarray
    name: str = "evaluator"

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != (N_FEATURES,):
            raise ValueError(
                f"weights must have shape ({N_FEATURES},), got {self.weights.shape}"
            )

    def score(self, board: chess.Board) -> float:
        return float(self.weights @ phi(board))

    def score_for_side_to_move(self, board: chess.Board) -> float:
        s = self.score(board)
        return s if board.turn == chess.WHITE else -s

    @classmethod
    def zero(cls) -> "Evaluator":
        return cls(np.zeros(N_FEATURES, dtype=np.float32), name="zero")

    @classmethod
    def material_only(cls) -> "Evaluator":
        w = np.zeros(N_FEATURES, dtype=np.float32)
        w[MATERIAL_SLICE] = [
            MATERIAL[chess.PAWN],
            MATERIAL[chess.KNIGHT],
            MATERIAL[chess.BISHOP],
            MATERIAL[chess.ROOK],
            MATERIAL[chess.QUEEN],
        ]
        w[BISHOP_PAIR_INDEX] = 30.0
        w[SIDE_TO_MOVE_INDEX] = 10.0
        return cls(w, name="material_only")

    @classmethod
    def psqt_handcrafted(cls) -> "Evaluator":
        w = cls.material_only().weights.copy()
        for pt, sl in PSQT_SLICES.items():
            w[sl] = PSQT[pt - 1].reshape(64).astype(np.float32)
        return cls(w, name="psqt_handcrafted")
