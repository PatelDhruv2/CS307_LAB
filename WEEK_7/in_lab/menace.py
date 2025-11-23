import random

class Menace:
    def _init_(self):
        self.matchboxes = {}  # Matchboxes for each distinct state
        self.create_matchboxes()

    def rotate_state(self, state):
        return (state[6], state[3], state[0],
                state[7], state[4], state[1],
                state[8], state[5], state[2])

    def reflect_state(self, state):
        return (state[2], state[1], state[0],
                state[5], state[4], state[3],
                state[8], state[7], state[6])

    def get_all_symmetries(self, state):
        symmetries = []
        rotated = state
        for _ in range(4):  # Rotate 0, 90, 180, 270 degrees
            rotated = self.rotate_state(rotated)
            symmetries.append(rotated)
            symmetries.append(self.reflect_state(rotated))  # Add reflections of each rotation
        return symmetries

    def get_canonical_state(self, state):
        symmetries = self.get_all_symmetries(state)
        return min(symmetries)

    def create_matchboxes(self):
        for state in self.all_possible_states():
            canonical_state = self.get_canonical_state(state)
            if canonical_state not in self.matchboxes:
                self.matchboxes[canonical_state] = self.create_beads(canonical_state)

    def all_possible_states(self):
        states = []
        for i in range(3**9):  # There are 3^9 possible states
            board = self.int_to_state(i)
            states.append(board)
        return states

    def int_to_state(self, n):
        board = []
        for i in range(9):
            board.append(n % 3)  # 0 = empty, 1 = 'X', 2 = 'O'
            n //= 3
        return tuple(board)

    def create_beads(self, state):
        beads = {}
        for i, cell in enumerate(state):
            if cell == 0:  # If the cell is empty
                beads[i] = 1  # 1 bead for each possible move
        return beads

    def choose_move(self, state):
        canonical_state = self.get_canonical_state(state)
        beads = self.matchboxes[canonical_state]
        total_beads = sum(beads.values())
        r = random.randint(1, total_beads)
        cumulative = 0
        for move, count in beads.items():
            cumulative += count
            if r <= cumulative:
                return move

    def update_beads(self, state, move, result):
        canonical_state = self.get_canonical_state(state)

        # Check if the move is valid for the current state
        if move not in self.matchboxes[canonical_state]:
            print(f"Invalid move {move} for state {canonical_state}. Current beads: {self.matchboxes[canonical_state]}")
            return  # Exit if the move is not valid

        if result == "win":
            self.matchboxes[canonical_state][move] += 3  # Add 3 beads for a win
        elif result == "draw":
            self.matchboxes[canonical_state][move] += 1  # Add 1 bead for a draw
        elif result == "loss":
            self.matchboxes[canonical_state][move] = max(1, self.matchboxes[canonical_state][move] - 1)  # Remove 1 bead for a loss

    def play_game(self, opponent):
        game_history = []
        state = (0,) * 9  # Empty board
        for turn in range(9):
            if turn % 2 == 0:
                move = self.choose_move(state)
            else:
                move = opponent.choose_move(state)
            game_history.append((state, move))
            state = self.make_move(state, move, turn % 2)
            result = self.check_winner(state)
            if result:
                self.update_game_history(game_history, result)
                return result

    def make_move(self, state, move, player):
        new_state = list(state)
        new_state[move] = 1 if player == 0 else 2
        return tuple(new_state)

    def check_winner(self, state):
        winning_positions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
                             (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
                             (0, 4, 8), (2, 4, 6)]  # Diagonals
        for pos in winning_positions:
            if state[pos[0]] == state[pos[1]] == state[pos[2]] != 0:
                return "win" if state[pos[0]] == 1 else "loss"
        if 0 not in state:
            return "draw"
        return None

    def update_game_history(self, game_history, result):
        j = 0
        decay = 1
        for state, move in reversed(game_history):
            j += 1
            self.update_beads(state, move, result)  # Call without additional argument
            decay *= 0.9


class RandomOpponent:
    def choose_move(self, state):
        empty_cells = [i for i, cell in enumerate(state) if cell == 0]
        return random.choice(empty_cells)

# Simulating a few games
menace = Menace()
opponent = RandomOpponent()

for _ in range(1000):
    result = menace.play_game(opponent)
    print(f"Game result: {result}")