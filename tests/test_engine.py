"""Sanity tests. Run with `pytest -q`."""

from __future__ import annotations

import math

import chess
import numpy as np
import pytest

from chess_engine.evaluate import Evaluator
from chess_engine.features import (
    BISHOP_PAIR_INDEX,
    MATERIAL_SLICE,
    N_FEATURES,
    PSQT_SLICES,
    SIDE_TO_MOVE_INDEX,
    phi,
)
from chess_engine.search import negamax_alphabeta, search_best_move


def test_phi_shape_and_dtype():
    v = phi(chess.Board())
    assert v.shape == (N_FEATURES,)
    assert v.dtype == np.float32


def test_phi_starting_position_is_symmetric():
    v = phi(chess.Board())
    assert np.all(v[MATERIAL_SLICE] == 0)
    for sl in PSQT_SLICES.values():
        assert np.all(v[sl] == 0), f"non-zero PSQT diff in slice {sl}"
    assert v[BISHOP_PAIR_INDEX] == 0
    assert v[SIDE_TO_MOVE_INDEX] == 1.0


def test_phi_after_moving_one_pawn():
    board = chess.Board()
    board.push_san("e4")
    v = phi(board)
    assert np.all(v[MATERIAL_SLICE] == 0)
    pawn_diff = v[PSQT_SLICES[chess.PAWN]]
    # e2 now empty (was +1), e4 now occupied (+1) -> two nonzero entries.
    nonzero = np.flatnonzero(pawn_diff)
    assert len(nonzero) == 2
    assert v[SIDE_TO_MOVE_INDEX] == -1.0


def test_evaluator_starting_position_is_balanced():
    # material_only() at the start should give just the tempo bonus.
    e = Evaluator.material_only()
    score = e.score(chess.Board())
    assert score == pytest.approx(10.0)


def test_evaluator_white_up_a_queen():
    # standard start, but black has no queen on d8
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    e = Evaluator.material_only()
    assert e.score(board) > 900   # ~+1025 from the queen plus tempo


def test_search_captures_hanging_queen():
    # white rook a1, undefended black queen a8 -> Rxa8 wins instantly
    board = chess.Board("q6k/8/8/8/8/8/8/R6K w - - 0 1")
    e = Evaluator.material_only()
    result = search_best_move(board, depth=1, evaluator=e)
    assert result.move == chess.Move.from_uci("a1a8")


def test_alphabeta_matches_plain_minimax():
    board = chess.Board()
    e = Evaluator.psqt_handcrafted()
    s_plain, n_plain = negamax_alphabeta(board, depth=3, evaluator=e, prune=False, ordered=False)
    s_ab,    n_ab    = negamax_alphabeta(board, depth=3, evaluator=e, prune=True,  ordered=False)
    s_ab_o,  n_ab_o  = negamax_alphabeta(board, depth=3, evaluator=e, prune=True,  ordered=True)
    assert math.isclose(s_plain, s_ab, abs_tol=1e-3)
    assert math.isclose(s_plain, s_ab_o, abs_tol=1e-3)
    assert n_ab <= n_plain
    assert n_ab_o <= n_plain


def test_search_detects_mate_in_one():
    # smothered-mate-ish: black Kh8 boxed in by Rg8, Pg7, Ph7. Nh6-f7#.
    board = chess.Board("6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1")
    e = Evaluator.material_only()
    result = search_best_move(board, depth=2, evaluator=e)
    assert result.move == chess.Move.from_uci("h6f7")
    assert result.score > 100_000
