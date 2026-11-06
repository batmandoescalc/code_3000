import sys

class TicTacToe:

    def __init__(self):
        self.diag = 0
        self.anti_diag = 0
        self.cols = [0 for _ in range(3)]
        self.rows = [0 for _ in range(3)]
        self.matrix = ['_' for _ in range(9)]

    def make_move(self, player, col, row) -> int:

        self.cols[col] += player
        self.rows[row] += player
        if row == col: self.diag += player
        if row + col == 2: self.anti_diag += player

        self.matrix[col + (3 * row)] = 'X' if player == 1 else 'O'
        if self.diag == 3 or self.diag == -3:
            return player
        if self.anti_diag == 3 or self.anti_diag == -3:
            return player
        for x in range(3):
            if self.cols[x] == 3 or self.rows[x] == 3 or self.cols[x] == -3 or self.rows[x] == -3:
                return player

        return 0
    def print_matrix(self):
        for x in range(len(self.matrix)):
            row = (x + 1)%3
            if row == 1: 
                print("[" + self.matrix[x], end='')
            elif row == 2:
                print("|" + self.matrix[x] + "|", end='')
            else:
                print(self.matrix[x] + "]")

if __name__ == "__main__":
    TTT = TicTacToe()

    gamestate = 0
    currentPlayer = 1
    TTT.print_matrix()
    while(gamestate == 0):
        col, row = tuple(int(x) for x in input("Enter col and row: ").split())
        if TTT.matrix[col + 3*row] == '_':
            gamestate = TTT.make_move(currentPlayer, col, row)
            TTT.print_matrix()
            currentPlayer = currentPlayer * -1
        else:
            sys.stdout.write("\033[1A\033[2K")
            sys.stdout.flush()
            print("INVALID MOVE: ", end= '')
            continue
    if gamestate == 1: print("X wins.")
    if gamestate == -1: print("O wins.")   