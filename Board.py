import chess
class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board = chess.Board()

    def get_legal_moves(self):
        return [str(move) for move in self.board.legal_moves]

    def make_move(self, move):
        self.board.push_uci(move)