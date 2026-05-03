"""Feature vector phi(board) for the linear evaluation w . phi(board).

Layout (length 391, dtype float32, white's POV):
    [0:5]       P, N, B, R, Q material count, white minus black
    [5:69]      pawn   PSQT indicator: white_bb minus flipud(black_bb)
    [69:133]    knight PSQT
    [133:197]   bishop PSQT
    [197:261]   rook   PSQT
    [261:325]   queen  PSQT
    [325:389]   king   PSQT
    [389]       bishop pair, signed
    [390]       side to move, +1 / -1 (tempo)
"""

from __future__ import annotations

import chess
import numpy as np

_PIECE_TYPES_PSQT = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
_PIECE_LETTERS = ("P", "N", "B", "R", "Q", "K")

MATERIAL_SLICE = slice(0, 5)
PSQT_SLICES: dict[int, slice] = {
    chess.PAWN:   slice(5,   69),
    chess.KNIGHT: slice(69,  133),
    chess.BISHOP: slice(133, 197),
    chess.ROOK:   slice(197, 261),
    chess.QUEEN:  slice(261, 325),
    chess.KING:   slice(325, 389),
}
BISHOP_PAIR_INDEX = 389
SIDE_TO_MOVE_INDEX = 390
N_FEATURES = 391


def _build_feature_names() -> list[str]:
    names: list[str] = [f"material_{p}" for p in "PNBRQ"]
    for letter in _PIECE_LETTERS:
        for sq in range(64):
            names.append(f"psqt_{letter}_{chess.square_name(sq)}")
    names.append("bishop_pair")
    names.append("side_to_move")
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
assert len(FEATURE_NAMES) == N_FEATURES


def _bb_to_array(bb: int) -> np.ndarray:
    bytes_ = np.array([bb], dtype="<u8").view(np.uint8)
    bits = np.unpackbits(bytes_, bitorder="little")
    return bits.reshape(8, 8)


def phi(board: chess.Board) -> np.ndarray:
    out = np.zeros(N_FEATURES, dtype=np.float32)

    for i, pt in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
        out[i] = (
            chess.popcount(board.pieces_mask(pt, chess.WHITE))
            - chess.popcount(board.pieces_mask(pt, chess.BLACK))
        )

    for pt in _PIECE_TYPES_PSQT:
        white_arr = _bb_to_array(board.pieces_mask(pt, chess.WHITE))
        black_arr = _bb_to_array(board.pieces_mask(pt, chess.BLACK))
        diff = white_arr.astype(np.int8) - np.flipud(black_arr).astype(np.int8)
        out[PSQT_SLICES[pt]] = diff.reshape(64)

    white_bishops = chess.popcount(board.pieces_mask(chess.BISHOP, chess.WHITE))
    black_bishops = chess.popcount(board.pieces_mask(chess.BISHOP, chess.BLACK))
    out[BISHOP_PAIR_INDEX] = (1 if white_bishops >= 2 else 0) - (1 if black_bishops >= 2 else 0)

    out[SIDE_TO_MOVE_INDEX] = 1.0 if board.turn == chess.WHITE else -1.0

    return out
