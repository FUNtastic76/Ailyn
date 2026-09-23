import pygame
import random
from Ailyn_Brain.visual_cortex import VisualCortex
from Ailyn_Brain.Ailyn_central_system import AilynCore
from Ailyn_Brain.limbic_system import LimbicSystem
import numpy as np
import multiprocessing
import queue

# --- parameters ---
BLOCK_SIZE = 30
COLS = 10
ROWS = 20
FPS = 30
FALL_SPEED = 30

# --- colours ---
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
RED = (220, 50, 50)
GREEN = (50, 220, 50)
BLUE = (50, 50, 220)
CYAN = (50, 220, 220)
MAGENTA = (220, 50, 220)
YELLOW = (220, 220, 50)
ORANGE = (255, 165, 0)

COLORS = [BLACK, CYAN, BLUE, ORANGE, YELLOW, GREEN, MAGENTA, RED]

# --- objects ---
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[0, 1, 0], [1, 1, 1]],  # T
    [[1, 1, 0], [0, 1, 1]]  # Z
]


def ailyn_brain_multiprocess(vision_queue, action_queue):
    import traceback

    try:
        brain = AilynCore()
        limbic = LimbicSystem()
        old_score = 0

        brain.focus_x = 550.0
        brain.focus_y = 350.0

        action_trace = np.zeros(5)
        trace_decay = 0.9

    except Exception as e:
        print(f"[FATAL] Ошибка мозга: {e}")
        return

    while True:
        try:
            data = vision_queue.get()
            if data == "SLEEP":
                break

            retina_list, step_reward, piece_info = data
            piece_x, piece_y, piece_w, piece_h = piece_info
            retina_signal = np.array(retina_list)

            brain.predictive_programming(retina_signal)
            is_watching_player = brain.focus_x < 400

            limbic.amygdala(brain)

            # eligibility trace
            if step_reward != 0:
                if is_watching_player:
                    step_reward = -1.0

                for a in range(len(action_trace)):
                    specific_reward = step_reward * action_trace[a]
                    limbic.dopamine_release(brain, specific_reward)

            import math
            BOARD_OFFSET_X, BOARD_OFFSET_Y = 400, 50
            piece_pixel_x = BOARD_OFFSET_X + (piece_x + piece_w / 2) * BLOCK_SIZE
            piece_pixel_y = BOARD_OFFSET_Y + (piece_y + piece_h / 2) * BLOCK_SIZE

            stack_pixel_y = BOARD_OFFSET_Y + (ROWS * BLOCK_SIZE)

            #target_gaze_y = (piece_pixel_y + stack_pixel_y) / 2.0

            #dist_to_target: float | int = math.sqrt((brain.focus_x - piece_pixel_x) ** 2 + (brain.focus_y - target_gaze_y) ** 2)

            #if dist_to_target > 200.0:
             #   step_reward -= 1.0
             #   brain.Total_energy += 0.5

            #gaze_reward = math.exp(-dist_to_target / 150.0) * 0.4 - 0.1
            #limbic.dopamine_release(brain, gaze_reward)

            # Solutions
            hands_probs, eyes_probs = brain.get_actions()
            hands_probs = np.array(hands_probs, dtype=np.float64)
            prob_sum = np.sum(hands_probs)

            # Normazilization
            if prob_sum > 0:
                hands_probs /= prob_sum
            else:
                hands_probs = np.ones(len(hands_probs)) / len(hands_probs)

            hand_action = int(np.random.choice(len(hands_probs), p=hands_probs))

            action_trace *= trace_decay # forget old memory
            action_trace[hand_action] += 1.0 # remember current

            #brain.move_eyes(eyes_probs)

            #brain.focus_x = 550.0

            #brain.focus_y = max(250.0, min(brain.focus_y, 450.0))

            brain.focus_x = float(piece_pixel_x)
            brain.focus_y = float(piece_pixel_y)

            response = {
                'hand_action': hand_action,
                'focus_x': float(brain.focus_x),
                'focus_y': float(brain.focus_y),
                'energy': float(brain.Total_energy),
                'arousal': float(limbic.arousal),
                'valence': float(limbic.valence),
                'emotion': limbic.state,
                'lr': float(brain.learning_rate)
            }
            action_queue.put(response)

        except Exception as e:
            print("\n!!! ОШИБКА В ЯДРЕ !!!")
            traceback.print_exc()
            dummy = {'hand_action': 4, 'focus_x': 750.0, 'focus_y': 350.0, 'energy': -99.0, 'arousal': 0.0,
                     'valence': 0.0, 'emotion': "ERROR", 'lr': 0.0}
            action_queue.put(dummy)

class Tetris:
    def __init__(self, use_curriculum=True):
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.score = 0
        self.game_over = False
        self.spawn_piece()
        self.step_reward = 0.0
        self.last_step_reward = 0.0

        if use_curriculum:
            self.setup_curriculum_board()

        self.spawn_piece()

    def spawn_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        self.current_piece = SHAPES[shape_idx]
        self.current_color = shape_idx + 1
        self.piece_x = COLS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0

        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def setup_curriculum_board(self):
        hole_start = random.randint(0, COLS - 2)

        for col in range(COLS):
            if col != hole_start and col != hole_start + 1:
                self.grid[ROWS - 1][col] = 1


        for col in range(COLS):
            if self.grid[ROWS - 1][col] != 0:
                if random.random() < 0.5:
                    self.grid[ROWS - 2][col] = 1

    def check_collision(self, shape, offset_x, offset_y):
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell:
                    x = cx + offset_x
                    y = cy + offset_y
                    if x < 0 or x >= COLS or y >= ROWS:
                        return True
                    if y >= 0 and self.grid[y][x]:
                        return True
        return False

    def rotate(self):
        rotated = [list(row) for row in zip(*self.current_piece[::-1])]
        if not self.check_collision(rotated, self.piece_x, self.piece_y):
            self.current_piece = rotated

    def move(self, dx, dy):
        if not self.check_collision(self.current_piece, self.piece_x + dx, self.piece_y + dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        return False

    def drop(self):
        if not self.move(0, 1):
            self.lock_piece()

    def hard_drop(self):
        while self.move(0, 1):
            pass
        self.lock_piece()

    def lock_piece(self):
        for cy, row in enumerate(self.current_piece):
            for cx, cell in enumerate(row):
                if cell:
                    self.grid[self.piece_y + cy][self.piece_x + cx] = self.current_color

        # --- reward calculation ---
        holes = 0
        heights = [0] * COLS

        # calculate number of holes
        for col in range(COLS):
            block_found = False
            for row in range(ROWS):
                if self.grid[row][col] != 0:
                    if not block_found:
                        heights[col] = ROWS - row
                        block_found = True
                elif block_found and self.grid[row][col] == 0:
                    holes += 1

        aggregate_height = sum(heights)

        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1))

        # rewarding and penalizing
        self.step_reward += 0.2
        self.step_reward -= (holes * 4.0)
        self.step_reward -= (bumpiness * 1.5)
        self.step_reward -= (aggregate_height * 0.2)

        self.score += 10 # for locking a piece

        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = ROWS - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, [0 for _ in range(COLS)])
        self.grid = new_grid
        self.score += lines_cleared * 100

        if lines_cleared == 1:
            self.step_reward += 20.0
        elif lines_cleared == 2:
            self.step_reward += 50.0
        elif lines_cleared >= 3:
            self.step_reward += 100.0

def draw_board(surface, game, offset_x, offset_y, title):
    font = pygame.font.SysFont('Arial', 24, bold=True)
    title_surface = font.render(title, True, WHITE)
    surface.blit(title_surface, (offset_x + 10, offset_y - 35))

    for y in range(ROWS):
        for x in range(COLS):
            color = COLORS[game.grid[y][x]]
            rect = (offset_x + x * BLOCK_SIZE, offset_y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRAY, rect, 1)  # Обводка

    if not game.game_over:
        for cy, row in enumerate(game.current_piece):
            for cx, cell in enumerate(row):
                if cell:
                    rect = (offset_x + (game.piece_x + cx) * BLOCK_SIZE,
                            offset_y + (game.piece_y + cy) * BLOCK_SIZE,
                            BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(surface, COLORS[game.current_color], rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)

    score_surface = font.render(f"Score: {game.score}", True, WHITE)
    surface.blit(score_surface, (offset_x, offset_y + ROWS * BLOCK_SIZE + 10))
    if game.game_over:
        over_surface = font.render("GAME OVER", True, RED)
        surface.blit(over_surface, (offset_x + 60, offset_y + ROWS * BLOCK_SIZE // 2))

# not using anymore
def ailyn_make_decision(game_state):

    actions = ['LEFT', 'RIGHT', 'ROTATE', 'DROP', 'NONE', 'NONE', 'NONE']
    return random.choice(actions)


def draw_ailyn_hud(surface, telemetry, last_reward, offset_x, offset_y):
    font = pygame.font.SysFont('Consolas', 18, bold=True)

    # Telemetry display with color coding based on emotion
    lines = [
        "=== КОГНИТИВНАЯ ПАНЕЛЬ АЙЛИН ===",
        f"Emotion State : {telemetry['emotion'].upper()}",
        f"Total Energy  : {telemetry['energy']:.2f} (Стресс)",
        f"Arousal       : {telemetry['arousal']:.3f}",
        f"Valence       : {telemetry['valence']:.3f}",
        f"Learning Rate : {telemetry['lr']:.6f}",
        f"Focus X       : {telemetry['focus_x']:.1f}",
        f"Focus Y       : {telemetry['focus_y']:.1f}",
        f"Step Reward   : {last_reward:+.2f}"
    ]

    for i, line in enumerate(lines):
        color = GREEN if "positive" in telemetry['emotion'] else (
            RED if "negative" in telemetry['emotion'] else YELLOW) if i == 1 else WHITE
        text_surface = font.render(line, True, color)
        surface.blit(text_surface, (offset_x, offset_y + i * 30))


def draw_crosshair(surface, x, y):
    color = (0, 255, 0)
    length = 15
    thickness = 2

    pygame.draw.line(surface, color, (x - length, y), (x + length, y), thickness)
    pygame.draw.line(surface, color, (x, y - length), (x, y + length), thickness)

    pygame.draw.circle(surface, color, (x, y), 2)
    pygame.draw.rect(surface, color, (x - 200, y - 200, 400, 400), 1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((1500, BLOCK_SIZE * ROWS + 100))
    pygame.display.set_caption("Tetris Arena: Player vs Ailyn")
    clock = pygame.time.Clock()

    game_human = Tetris()
    game_ailyn = Tetris()

    # --- Brain ---
    vision_queue = multiprocessing.Queue()
    action_queue = multiprocessing.Queue()

    brain_process = multiprocessing.Process(
        target=ailyn_brain_multiprocess,
        args=(vision_queue, action_queue)
    )
    brain_process.start()

    ailyn_telemetry = {
        'hand_action': 4, 'focus_x': 550.0, 'focus_y': 350.0,
        'energy': 0.0, 'arousal': 0.0, 'valence': 0.0,
        'emotion': "neutral", 'lr': 0.0
    }

    ai_is_thinking = False


    ailyn_eye = VisualCortex()
    '''
    ailyn_brain = AilynCore()
    ailyn_limbic = LimbicSystem()

    ailyn_brain.focus_x = float(BLOCK_SIZE * COLS + 100 + (BLOCK_SIZE * COLS) // 2)
    ailyn_brain.focus_y = float(50 + (BLOCK_SIZE * ROWS) // 2)
    '''

    frame_counter = 0

    ailyn_old_score = 0

    running = True
    while running:
        screen.fill(BLACK)
        frame_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Управление Человека
            if event.type == pygame.KEYDOWN:
                if not game_human.game_over:
                    if event.key == pygame.K_LEFT:
                        game_human.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        game_human.move(1, 0)
                    elif event.key == pygame.K_UP:
                        game_human.rotate()
                    elif event.key == pygame.K_DOWN:
                        game_human.move(0, 1)
                    elif event.key == pygame.K_SPACE:
                        game_human.hard_drop()

                if event.key == pygame.K_r:
                    game_human = Tetris()
                    game_ailyn = Tetris()
                    ailyn_old_score = 0
                    print("Айлин начинает новую попытку.")

                if event.key == pygame.K_m:
                    game_ailyn.score += 100
                    with open("D:/AI/PythonProject/Ailyn/milk_signal.txt", "w") as file:
                        file.write("1")

                if event.key == pygame.K_n:
                    game_ailyn.score -= 100
                    print("PENALTY: -1")

        # gravitation
        if frame_counter % FALL_SPEED == 0:
            if not game_human.game_over:
                game_human.drop()
            if not game_ailyn.game_over:
                game_ailyn.drop()

        draw_board(screen, game_human, offset_x=50, offset_y=50, title="Player")
        draw_board(screen, game_ailyn, offset_x=BLOCK_SIZE * COLS + 100, offset_y=50, title="Ailyn")

        if game_ailyn.game_over:
            final_penalty = game_ailyn.step_reward - 10.0

            with open("ailyn_training_log.txt", "a") as log:
                log.write(f"Score: {game_ailyn.score} | Energy: {ailyn_telemetry['energy']:.2f}\n")

            print(f"[NIGHT RUN] Game Over. Score: {game_ailyn.score}. Restarting...")

            pygame.time.delay(100)

            game_ailyn = Tetris()

            # she remembers her last step reward before game over, so we can use it for training
            game_ailyn.step_reward = final_penalty
            game_ailyn.last_step_reward = final_penalty

        if not ai_is_thinking and not game_ailyn.game_over:

            #full_screen = pygame.surfarray.array3d(screen)
            #matrix_form = full_screen.transpose(1, 0, 2)

            FOV_SIZE = 400
            half_fov = FOV_SIZE // 2

            piece_w = len(game_ailyn.current_piece[0])
            piece_h = len(game_ailyn.current_piece)
            BOARD_OFFSET_X = 400
            BOARD_OFFSET_Y = 50

            target_focus_x = BOARD_OFFSET_X + (game_ailyn.piece_x + piece_w / 2) * BLOCK_SIZE
            target_focus_y = BOARD_OFFSET_Y + (game_ailyn.piece_y + piece_h / 2) * BLOCK_SIZE

            # force focus to stay within the screen bounds
            ailyn_telemetry['focus_x'] = target_focus_x
            ailyn_telemetry['focus_y'] = target_focus_y

            focus_x = int(ailyn_telemetry['focus_x'])
            focus_y = int(ailyn_telemetry['focus_y'])

            min_x = max(0, focus_x - half_fov)
            max_x = min(screen.get_width(), focus_x + half_fov)
            min_y = max(0, focus_y - half_fov)
            max_y = min(screen.get_height(), focus_y + half_fov)

            fov_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            fov_surface = screen.subsurface(fov_rect)

            ailyn_fov = pygame.surfarray.array3d(fov_surface).transpose(1, 0, 2)

            #ailyn_fov = matrix_form[min_y:max_y, min_x:max_x]

            if ailyn_fov.shape[0] == FOV_SIZE and ailyn_fov.shape[1] == FOV_SIZE:
                retina_signal = ailyn_eye.process_vision(ailyn_fov)

                piece_w = len(game_ailyn.current_piece[0])
                piece_h = len(game_ailyn.current_piece)
                piece_info = (game_ailyn.piece_x, game_ailyn.piece_y, piece_w, piece_h)
                vision_queue.put((retina_signal.tolist(), game_ailyn.step_reward, piece_info))

                game_ailyn.last_step_reward = game_ailyn.step_reward

                game_ailyn.step_reward = 0.0

                ai_is_thinking = True

        try:
            response = action_queue.get_nowait()
            ailyn_telemetry = response
            ai_is_thinking = False

            is_watching_player = ailyn_telemetry['focus_x'] < 400
            if not is_watching_player:
                action = ailyn_telemetry['hand_action']
                if action == 0:
                    game_ailyn.move(-1, 0)
                elif action == 1:
                    game_ailyn.move(1, 0)
                elif action == 2:
                    game_ailyn.rotate()
                elif action == 3:
                    if not game_ailyn.move(0, 1):
                        game_ailyn.lock_piece()

        except queue.Empty:
            pass

        draw_ailyn_hud(screen, ailyn_telemetry, game_ailyn.last_step_reward, offset_x=750, offset_y=50)

        draw_crosshair(screen, int(ailyn_telemetry['focus_x']), int(ailyn_telemetry['focus_y']))

        pygame.display.flip()
        clock.tick(FPS)

    vision_queue.put("SLEEP")
    brain_process.join()
    pygame.quit()

    '''
    retina_signal = ailyn_eye.process_vision(ailyn_fov)

    ailyn_brain.predictive_programming(retina_signal)
    ailyn_limbic.amygdala(ailyn_brain)

    if game_ailyn.score > ailyn_old_score:
        reward = (game_ailyn.score - ailyn_old_score) / 100.0  # Масштабируем награду
        ailyn_limbic.dopamine_release(ailyn_brain, reward)
        ailyn_old_score = game_ailyn.score

    hands_probs, eyes_probs = ailyn_brain.get_actions()

    if not is_watching_player:
        hand_action = np.argmax(hands_probs)
        if hand_action == 0:
            game_ailyn.move(-1, 0)
        elif hand_action == 1:
            game_ailyn.move(1, 0)
        elif hand_action == 2:
            game_ailyn.rotate()
        elif hand_action == 3:
            game_ailyn.move(0, 1)

    ailyn_brain.move_eyes(eyes_probs)
    '''

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()