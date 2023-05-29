import numpy as np

class GameBot:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.board = [' ' for _ in range(9)]
        self.player_symbol = 'X'
        self.bot_symbol = 'O'
        self.q_values = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def start_game(self):
        self.board = [' ' for _ in range(9)]
        self.player_symbol = 'X'
        self.bot_symbol = 'O'
        print("Welcome to the Tic-Tac-Toe Game!")
        self.display_board()

    def display_board(self):
        print('-------------')
        for i in range(3):
            print('|', self.board[i * 3], '|', self.board[i * 3 + 1], '|', self.board[i * 3 + 2], '|')
            print('-------------')

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.player_symbol
            self.display_board()

            if self.check_winner(self.player_symbol):
                return True, 1
            elif self.check_tie():
                return True, 0

            bot_move = self.bot_move()
            self.board[bot_move] = self.bot_symbol
            self.display_board()

            if self.check_winner(self.bot_symbol):
                return True, -1
            elif self.check_tie():
                return True, 0
        else:
            print("Invalid move. Try again.")
        return False, 0

    def bot_move(self):
        available_moves = [i for i, spot in enumerate(self.board) if spot == ' ']

        if np.random.uniform(0, 1) < self.epsilon:
            move = np.random.choice(available_moves)
        else:
            q_values = [self.q_values.get((tuple(self.board), move), 0) for move in available_moves]
            max_q = np.max(q_values)
            best_moves = [move for move, q_value in zip(available_moves, q_values) if q_value == max_q]
            move = np.random.choice(best_moves)

        return move

    def update_q_values(self, prev_state, prev_move, new_state, reward):
        prev_q_value = self.q_values.get((prev_state, prev_move), 0)
        max_q_value = self.get_max_q_value(new_state)
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * max_q_value - prev_q_value)
        self.q_values[(prev_state, prev_move)] = new_q_value

    def get_max_q_value(self, state):
        available_moves = [i for i, spot in enumerate(state) if spot == ' ']
        q_values = [self.q_values.get((state, move), 0) for move in available_moves]
        if q_values:
            return np.max(q_values)
        else:
            return 0

    def check_winner(self, symbol):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for combination in winning_combinations:
            if all(self.board[i] == symbol for i in combination):
                return True
        return False

    def check_tie(self):
        return ' ' not in self.board

# Function to play multiple games for training
def play_games(bot, num_games):
    for _ in range(num_games):
        bot.start_game()
        game_over = False
        prev_state = None
        prev_move = None

        while not game_over:
            player_move = int(input("Enter your move (0-8): "))
            game_over, reward = bot.make_move(player_move)
            if prev_state is not None:
                bot.update_q_values(prev_state, prev_move, tuple(bot.board), reward)
            if not game_over:
                prev_state = tuple(bot.board)
                prev_move = bot.bot_move()

# Example usage
bot = GameBot()
play_games(bot, 10000)  # Play 10000 games to train the bot

# Save the trained Q-values to a file
np.save('q_values.npy', bot.q_values)
