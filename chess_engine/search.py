"""Negamax with optional alpha-beta pruning and MVV-LVA ordering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess

from chess_engine.evaluate import Evaluator
from chess_engine.psqt import MATERIAL

MATE_SCORE = 1_000_000


@dataclass
class SearchResult:
    move: chess.Move | None
    score: float
    nodes: int
    depth: int


def _terminal_score(board: chess.Board, plies_from_root: int) -> float | None:
    if board.is_checkmate():
        sign = -1 if board.turn == chess.WHITE else 1
        return sign * (MATE_SCORE - plies_from_root)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    ):
        return 0.0
    return None


def _mvv_lva_key(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        victim_val = MATERIAL[chess.PAWN]
    else:
        captured = board.piece_at(move.to_square)
        victim_val = MATERIAL[captured.piece_type] if captured else 0
    attacker = board.piece_at(move.from_square)
    attacker_val = MATERIAL[attacker.piece_type] if attacker else 0
    return 10_000 + victim_val - attacker_val


def _ordered_moves(board: chess.Board, ordered: bool) -> list[chess.Move]:
    moves = list(board.legal_moves)
    if ordered:
        moves.sort(key=lambda m: _mvv_lva_key(board, m), reverse=True)
    return moves


@dataclass
class _SearchState:
    nodes: int = 0


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    evaluator: Evaluator,
    plies_from_root: int,
    prune: bool,
    ordered: bool,
    state: _SearchState,
) -> float:
    state.nodes += 1

    term = _terminal_score(board, plies_from_root)
    if term is not None:
        return term if board.turn == chess.WHITE else -term

    if depth == 0:
        return evaluator.score_for_side_to_move(board)

    best = -math.inf
    for move in _ordered_moves(board, ordered):
        board.push(move)
        score = -_negamax(
            board,
            depth - 1,
            -beta,
            -alpha,
            evaluator,
            plies_from_root + 1,
            prune,
            ordered,
            state,
        )
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if prune and alpha >= beta:
            break
    return best


def negamax_alphabeta(
    board: chess.Board,
    depth: int,
    evaluator: Evaluator,
    *,
    prune: bool = True,
    ordered: bool = True,
) -> tuple[float, int]:
    state = _SearchState()
    score = _negamax(
        board,
        depth,
        -math.inf,
        math.inf,
        evaluator,
        plies_from_root=0,
        prune=prune,
        ordered=ordered,
        state=state,
    )
    return score, state.nodes


def search_best_move(
    board: chess.Board,
    depth: int,
    evaluator: Evaluator,
    *,
    prune: bool = True,
    ordered: bool = True,
) -> SearchResult:
    state = _SearchState()
    best_move: chess.Move | None = None
    best_score = -math.inf
    alpha, beta = -math.inf, math.inf

    for move in _ordered_moves(board, ordered):
        board.push(move)
        score = -_negamax(
            board,
            depth - 1,
            -beta,
            -alpha,
            evaluator,
            plies_from_root=1,
            prune=prune,
            ordered=ordered,
            state=state,
        )
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
        if prune and alpha >= beta:
            break

    white_pov = best_score if board.turn == chess.WHITE else -best_score
    return SearchResult(move=best_move, score=white_pov, nodes=state.nodes, depth=depth)
