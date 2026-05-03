"""Two evaluators play one game and we get a chess.pgn.Game back."""

from __future__ import annotations

import chess
import chess.pgn

from chess_engine.evaluate import Evaluator
from chess_engine.search import search_best_move


def play_game(
    white: Evaluator,
    black: Evaluator,
    *,
    depth: int = 2,
    max_plies: int = 200,
    starting_fen: str | None = None,
) -> chess.pgn.Game:
    board = chess.Board(starting_fen) if starting_fen else chess.Board()

    game = chess.pgn.Game()
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    game.headers["Event"] = "self-play"
    if starting_fen:
        game.setup(board)
    node: chess.pgn.GameNode = game

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        evaluator = white if board.turn == chess.WHITE else black
        result = search_best_move(board, depth, evaluator)
        if result.move is None:
            break
        board.push(result.move)
        node = node.add_variation(result.move)

    if board.is_game_over(claim_draw=True):
        game.headers["Result"] = board.result(claim_draw=True)
    else:
        game.headers["Result"] = "1/2-1/2"
        game.headers["Termination"] = "adjudication: ply cap"

    return game
