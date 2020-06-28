"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    flat_board = [item for row in board for item in row]
    empty_count = flat_board.count(EMPTY)

    return X if (empty_count % 2) != 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    return set((i, j) for i in range(0, 3) for j in range(0, 3) if board[i][j] == EMPTY)


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    i, j = action
    board_copy = copy.deepcopy(board)

    if board[i][j] != EMPTY:
        raise BaseException(f'Invalid Move {action}')

    board_copy[i][j] = player(board)

    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(0, 3):
        # Check rows
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]

        # Check columns
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not EMPTY:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]

    if board[2][0] == board[1][1] == board[0][2]:
        return board[2][0]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) is not None:
        return True
    elif [item for row in board for item in row].count(EMPTY) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 if its a tie.
    """

    w = winner(board)
    return 1 if w == X else (-1 if w == O else 0)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None

    p = player(board)
    best_move = ()

    alpha = -math.inf
    beta = math.inf
    depth = 5

    if p == O:
        # minimizing
        best_score = math.inf

        # start recursion by finding the best (O is trying to minimize,
        # so the opponent is trying to maximize) possible moves for the
        # opponent
        for action in actions(board):
            score = max_value(result(board, action), alpha, beta, depth)
            beta = min(score, best_score)

            # and keep track of the one that minimizes his score (in this case, minimizing his score means
            # tracking the lowest score, since the opponent tries to find the highest score)
            if score < best_score:
                best_score = score
                best_move = action

    else:
        # maximizing
        best_score = -math.inf

        # start recursion by finding the best (X is trying to maximize,
        # so the opponent is trying to minimize) possible moves for the
        # opponent
        for action in actions(board):
            score = min_value(result(board, action), alpha, beta, depth)
            alpha = max(score, best_score)

            # and keep track of the one that minimizes his score (in this case, minimizing his score means
            # tracking the highest score, since the opponent tries to find the lowest score)
            if score > best_score:
                best_score = score
                best_move = action

    return best_move


def max_value(board, alpha, beta, depth) -> int:
    """
    Returns the maximum possible value or a state of actions
    """

    if terminal(board) or depth == 0:
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action), alpha, beta, depth - 1))

        # alpha-beta pruning
        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v


def min_value(board, alpha, beta, depth) -> int:
    """
    Returns the minimum possible value for a state of actions
    """

    if terminal(board) or depth == 0:
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action), alpha, beta, depth - 1))

        # alpha-beta pruning
        beta = min(beta, v)
        if alpha >= beta:
            break

    return v
