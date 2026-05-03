"""
Watch the final AlphaZero checkpoint play against a deterministic heuristic.
Run: python play_heuristic.py --delay-ms 500
"""

import argparse

import numpy as np
import pygame
import torch

from play import (
    BOARD_SIZE,
    DEVICE,
    PATH,
    NUM_SIMULATIONS,
    AlphaZeroNet,
    GomokuGame,
    mcts_search,
)


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
DEFAULT_DELAY_MS = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch AlphaZero MCTS play against a deterministic Gomoku heuristic."
    )
    parser.add_argument(
        "--delay-ms",
        "--delay",
        dest="delay_ms",
        type=int,
        default=DEFAULT_DELAY_MS,
        help=f"Pause between moves in milliseconds. Default: {DEFAULT_DELAY_MS}.",
    )
    parser.add_argument(
        "--az-color",
        "--alpha-zero-color",
        choices=("black", "white"),
        default="black",
        help="Side played by AlphaZero. Default: black.",
    )
    args = parser.parse_args()
    if args.delay_ms < 0:
        parser.error("--delay-ms must be 0 or greater")
    return args


def player_name(player):
    return "Black (X)" if player == 1 else "White (O)"


def agent_name(player, az_player, num_simulations):
    return f"AlphaZero {num_simulations} MCTS" if player == az_player else "Heuristic"


def find_immediate_win(game, player):
    board = game.board
    for action in game.legal_moves():
        r, c = divmod(action, BOARD_SIZE)
        board[r, c] = player
        is_win = game._check_winner_at(r, c) == player
        board[r, c] = 0
        if is_win:
            return action
    return None


def pattern_score(count, open_ends):
    if count >= 5:
        return 100_000
    if count == 4:
        return 12_000 if open_ends else 1_500
    if count == 3:
        return 2_000 if open_ends == 2 else 350 if open_ends == 1 else 0
    if count == 2:
        return 250 if open_ends == 2 else 60 if open_ends == 1 else 0
    return 10 if open_ends == 2 else 2 if open_ends == 1 else 0


def line_features(board, r, c, player, dr, dc):
    count = 1
    open_ends = 0

    for sign in (1, -1):
        nr = r + sign * dr
        nc = c + sign * dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == player:
            count += 1
            nr += sign * dr
            nc += sign * dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
            open_ends += 1

    return count, open_ends


def score_action(game, action, player):
    board = game.board
    r, c = divmod(action, BOARD_SIZE)
    board[r, c] = player

    score = 0
    for dr, dc in DIRECTIONS:
        count, open_ends = line_features(board, r, c, player, dr, dc)
        score += pattern_score(count, open_ends)

    centre = (BOARD_SIZE - 1) / 2
    score += 20 - 3 * (abs(r - centre) + abs(c - centre))

    board[r, c] = 0
    return score


def heuristic_move(game):
    current = game.current_player
    opponent = -current

    winning_move = find_immediate_win(game, current)
    if winning_move is not None:
        return winning_move, "winning move"

    blocking_move = find_immediate_win(game, opponent)
    if blocking_move is not None:
        return blocking_move, "blocks win"

    best_key = None
    best_action = None
    centre = (BOARD_SIZE - 1) / 2

    for action in game.legal_moves():
        r, c = divmod(action, BOARD_SIZE)
        own_score = score_action(game, action, current)
        opponent_score = score_action(game, action, opponent)
        distance = abs(r - centre) + abs(c - centre)
        key = (own_score + 0.85 * opponent_score, -distance, -action)
        if best_key is None or key > best_key:
            best_key = key
            best_action = action

    return int(best_action), "best heuristic score"


class DemoRenderer:
    def __init__(self, az_player, num_simulations):
        self.az_player = az_player
        self.num_simulations = num_simulations
        self.cell_size = 50
        self.margin = 40
        self.extra_height = 95
        self.stone_radius = self.cell_size // 2 - 3

        self.width = (BOARD_SIZE - 1) * self.cell_size + 2 * self.margin
        self.height = (BOARD_SIZE - 1) * self.cell_size + 2 * self.margin + self.extra_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"AlphaZero {self.num_simulations} MCTS vs Heuristic")
        self.small_font = pygame.font.Font(None, 20)

        self.bg_color = (220, 179, 92)
        self.line_color = (0, 0, 0)
        self.black_color = (30, 30, 30)
        self.white_color = (240, 240, 240)
        self.last_move_color = (220, 50, 50)
        self.text_color = (40, 40, 40)

    def render_fit_text(self, message, size=24, min_size=16):
        max_width = self.width - 2 * self.margin
        for font_size in range(size, min_size - 1, -1):
            font = pygame.font.Font(None, font_size)
            text = font.render(message, True, self.text_color)
            if text.get_width() <= max_width:
                return text
        return pygame.font.Font(None, min_size).render(message, True, self.text_color)

    def render(self, game, status=None):
        self.screen.fill(self.bg_color)

        for i in range(BOARD_SIZE):
            x = self.margin + i * self.cell_size
            y_start = self.margin
            y_end = self.margin + (BOARD_SIZE - 1) * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (x, y_start), (x, y_end), 1)

            y = self.margin + i * self.cell_size
            x_start = self.margin
            x_end = self.margin + (BOARD_SIZE - 1) * self.cell_size
            pygame.draw.line(self.screen, self.line_color, (x_start, y), (x_end, y), 1)

        for i in range(BOARD_SIZE):
            label = self.small_font.render(str(i), True, self.text_color)
            lx = self.margin + i * self.cell_size - label.get_width() // 2
            self.screen.blit(label, (lx, self.margin - 25))

            label = self.small_font.render(str(i), True, self.text_color)
            ly = self.margin + i * self.cell_size - label.get_height() // 2
            self.screen.blit(label, (self.margin - 25, ly))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = game.board[r, c]
                if piece == 0:
                    continue
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                color = self.black_color if piece == 1 else self.white_color
                pygame.draw.circle(self.screen, color, (x, y), self.stone_radius)
                if piece == -1:
                    pygame.draw.circle(self.screen, (150, 150, 150), (x, y), self.stone_radius, 2)

        if game.last_move is not None:
            lr, lc = divmod(game.last_move, BOARD_SIZE)
            lx = self.margin + lc * self.cell_size
            ly = self.margin + lr * self.cell_size
            pygame.draw.circle(self.screen, self.last_move_color, (lx, ly), self.stone_radius, 3)

        black_agent = agent_name(1, self.az_player, self.num_simulations).replace(" MCTS", "")
        white_agent = agent_name(-1, self.az_player, self.num_simulations).replace(" MCTS", "")
        matchup = f"Black: {black_agent} | White: {white_agent}"
        matchup_text = self.render_fit_text(matchup, size=23, min_size=16)
        self.screen.blit(matchup_text, (self.margin, self.height - 78))

        if status is None:
            if game.is_terminal():
                status = result_message(game, self.az_player, self.num_simulations)
            else:
                status = f"Move {game.move_count + 1}: {agent_name(game.current_player, self.az_player, self.num_simulations)} to move"
        status_text = self.render_fit_text(status, size=24, min_size=16)
        self.screen.blit(status_text, (self.margin, self.height - 45))

        pygame.display.flip()

    def close(self):
        pygame.quit()


def result_message(game, az_player, num_simulations):
    winner = game.winner()
    if winner == 0:
        return "Draw"
    winner_agent = agent_name(winner, az_player, num_simulations)
    return f"{winner_agent} wins as {player_name(winner)}"


def wait_for_click(renderer, game):
    renderer.render(game, status="Click anywhere to start")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                return True
        pygame.time.wait(30)


def wait_with_events(delay_ms):
    elapsed = 0
    while elapsed < delay_ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        step = min(30, delay_ms - elapsed)
        pygame.time.wait(step)
        elapsed += step
    return True


def main():
    args = parse_args()
    az_player = 1 if args.az_color == "black" else -1

    print(f"Loading model: {PATH}")
    net = AlphaZeroNet().to(DEVICE)
    net.load_state_dict(torch.load(PATH, map_location=DEVICE, weights_only=True))
    net.eval()
    print(f"Device: {DEVICE}")
    print(f"Mode: AlphaZero {NUM_SIMULATIONS} MCTS vs heuristic")
    print(f"AlphaZero side: {player_name(az_player)}")
    print(f"Move delay: {args.delay_ms} ms")

    renderer = DemoRenderer(az_player, NUM_SIMULATIONS)
    game = GomokuGame()

    if not wait_for_click(renderer, game):
        renderer.close()
        return

    while not game.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                return

        current_agent = agent_name(game.current_player, az_player, NUM_SIMULATIONS)
        renderer.render(game, status=f"{current_agent} thinking...")
        pygame.event.pump()

        if game.current_player == az_player:
            visits = mcts_search(game, net, NUM_SIMULATIONS)
            action = int(np.argmax(visits))
            reason = f"{NUM_SIMULATIONS} sims"
        else:
            action, reason = heuristic_move(game)

        r, c = divmod(action, BOARD_SIZE)
        print(f"{current_agent} plays ({r},{c}) [{reason}]")
        game.play(action)
        renderer.render(game, status=f"{current_agent} played ({r},{c})")

        if args.delay_ms > 0 and not wait_with_events(args.delay_ms):
            renderer.close()
            return

    final_status = result_message(game, az_player, NUM_SIMULATIONS)
    renderer.render(game, status=final_status)
    print(final_status)
    print("Press any key or close the window to exit.")

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.wait(30)

    renderer.close()


if __name__ == "__main__":
    main()
