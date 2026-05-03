"""PeSTO middlegame piece-square tables.

PSQT[pt - 1, rank, file] is indexed with rank 0 = white's first rank, so it
lines up with chess.square_rank. The raw arrays are written in printed
order (rank 8 first) and then flipped.
"""

from __future__ import annotations

import chess
import numpy as np

# PeSTO middlegame material values, centipawns.
MATERIAL: dict[int, int] = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}

# Printed orientation: row 0 = rank 8, row 7 = rank 1.
_PAWN_MG = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 98, 134,  61,  95,  68, 126,  34, -11],
    [ -6,   7,  26,  31,  65,  56,  25, -20],
    [-14,  13,   6,  21,  23,  12,  17, -23],
    [-27,  -2,  -5,  12,  17,   6,  10, -25],
    [-26,  -4,  -4, -10,   3,   3,  33, -12],
    [-35,  -1, -20, -23, -15,  24,  38, -22],
    [  0,   0,   0,   0,   0,   0,   0,   0],
], dtype=np.int16)

_KNIGHT_MG = np.array([
    [-167, -89, -34, -49,  61, -97, -15, -107],
    [ -73, -41,  72,  36,  23,  62,   7,  -17],
    [ -47,  60,  37,  65,  84, 129,  73,   44],
    [  -9,  17,  19,  53,  37,  69,  18,   22],
    [ -13,   4,  16,  13,  28,  19,  21,   -8],
    [ -23,  -9,  12,  10,  19,  17,  25,  -16],
    [ -29, -53, -12,  -3,  -1,  18, -14,  -19],
    [-105, -21, -58, -33, -17, -28, -19,  -23],
], dtype=np.int16)

_BISHOP_MG = np.array([
    [-29,   4, -82, -37, -25, -42,   7,  -8],
    [-26,  16, -18, -13,  30,  59,  18, -47],
    [-16,  37,  43,  40,  35,  50,  37,  -2],
    [ -4,   5,  19,  50,  37,  37,   7,  -2],
    [ -6,  13,  13,  26,  34,  12,  10,   4],
    [  0,  15,  15,  15,  14,  27,  18,  10],
    [  4,  15,  16,   0,   7,  21,  33,   1],
    [-33,  -3, -14, -21, -13, -12, -39, -21],
], dtype=np.int16)

_ROOK_MG = np.array([
    [ 32,  42,  32,  51,  63,   9,  31,  43],
    [ 27,  32,  58,  62,  80,  67,  26,  44],
    [ -5,  19,  26,  36,  17,  45,  61,  16],
    [-24, -11,   7,  26,  24,  35,  -8, -20],
    [-36, -26, -12,  -1,   9,  -7,   6, -23],
    [-45, -25, -16, -17,   3,   0,  -5, -33],
    [-44, -16, -20,  -9,  -1,  11,  -6, -71],
    [-19, -13,   1,  17,  16,   7, -37, -26],
], dtype=np.int16)

_QUEEN_MG = np.array([
    [-28,   0,  29,  12,  59,  44,  43,  45],
    [-24, -39,  -5,   1, -16,  57,  28,  54],
    [-13, -17,   7,   8,  29,  56,  47,  57],
    [-27, -27, -16, -16,  -1,  17,  -2,   1],
    [ -9, -26,  -9, -10,  -2,  -4,   3,  -3],
    [-14,   2, -11,  -2,  -5,   2,  14,   5],
    [-35,  -8,  11,   2,   8,  15,  -3,   1],
    [ -1, -18,  -9,  10, -15, -25, -31, -50],
], dtype=np.int16)

_KING_MG = np.array([
    [-65,  23,  16, -15, -56, -34,   2,  13],
    [ 29,  -1, -20,  -7,  -8,  -4, -38, -29],
    [ -9,  24,   2, -16, -20,   6,  22, -22],
    [-17, -20, -12, -27, -30, -25, -14, -36],
    [-49,  -1, -27, -39, -46, -44, -33, -51],
    [-14, -14, -22, -46, -44, -30, -15, -27],
    [  1,   7,  -8, -64, -43, -16,   9,   8],
    [-15,  36,  12, -54,   8, -28,  24,  14],
], dtype=np.int16)

# Indexed by piece_type - 1 (PAWN=1 -> 0, ..., KING=6 -> 5).
PSQT = np.stack(
    [
        np.flipud(_PAWN_MG),
        np.flipud(_KNIGHT_MG),
        np.flipud(_BISHOP_MG),
        np.flipud(_ROOK_MG),
        np.flipud(_QUEEN_MG),
        np.flipud(_KING_MG),
    ],
    axis=0,
)


def psqt_value(piece_type: int, color: bool, square: int) -> int:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if color == chess.BLACK:
        rank = 7 - rank
    return int(PSQT[piece_type - 1, rank, file])
