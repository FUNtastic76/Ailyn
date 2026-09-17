import pygame
import random
from Ailyn_Brain.visual_cortex import VisualCortex

# --- Настройки ---
BLOCK_SIZE = 30
COLS = 10
ROWS = 20
FPS = 60
FALL_SPEED = 30  # Чем меньше, тем быстрее падает

# --- Цвета ---
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

# --- Формы фигур ---
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[0, 1, 0], [1, 1, 1]],  # T
    [[1, 1, 0], [0, 1, 1]]  # Z
]


class Tetris:
    def __init__(self):
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.score = 0
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        self.current_piece = SHAPES[shape_idx]
        self.current_color = shape_idx + 1
        self.piece_x = COLS // 2 - len(self.current_piece[0]) // 2
        self.piece_y = 0

        if self.check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

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
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = ROWS - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, [0 for _ in range(COLS)])
        self.grid = new_grid
        self.score += lines_cleared * 100


def draw_board(surface, game, offset_x, offset_y, title):
    font = pygame.font.SysFont('Arial', 24, bold=True)
    title_surface = font.render(title, True, WHITE)
    surface.blit(title_surface, (offset_x + 10, offset_y - 35))

    # Рисуем сетку
    for y in range(ROWS):
        for x in range(COLS):
            color = COLORS[game.grid[y][x]]
            rect = (offset_x + x * BLOCK_SIZE, offset_y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRAY, rect, 1)  # Обводка

    # Рисуем текущую фигуру
    if not game.game_over:
        for cy, row in enumerate(game.current_piece):
            for cx, cell in enumerate(row):
                if cell:
                    rect = (offset_x + (game.piece_x + cx) * BLOCK_SIZE,
                            offset_y + (game.piece_y + cy) * BLOCK_SIZE,
                            BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(surface, COLORS[game.current_color], rect)
                    pygame.draw.rect(surface, WHITE, rect, 1)

    # Счет и статус
    score_surface = font.render(f"Score: {game.score}", True, WHITE)
    surface.blit(score_surface, (offset_x, offset_y + ROWS * BLOCK_SIZE + 10))
    if game.game_over:
        over_surface = font.render("GAME OVER", True, RED)
        surface.blit(over_surface, (offset_x + 60, offset_y + ROWS * BLOCK_SIZE // 2))


# --- ЗАГЛУШКА ДЛЯ МОЗГА АЙЛИН ---
def ailyn_make_decision(game_state):
    # Пока её нейронная сеть не подключена, она просто дергается как новорожденная
    actions = ['LEFT', 'RIGHT', 'ROTATE', 'DROP', 'NONE', 'NONE', 'NONE']
    return random.choice(actions)


def main():
    pygame.init()
    screen = pygame.display.set_mode((1500, BLOCK_SIZE * ROWS + 100))
    pygame.display.set_caption("Tetris Arena: Human vs Ailyn")
    clock = pygame.time.Clock()

    game_human = Tetris()
    game_ailyn = Tetris()

    ailyn_eye = VisualCortex()
    focus_x = BLOCK_SIZE * COLS + 100 + (BLOCK_SIZE * COLS) // 2  # motor cortex
    focus_y = 50 + (BLOCK_SIZE * ROWS) // 2

    frame_counter = 0

    running = True
    while running:
        screen.fill(BLACK)
        frame_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Управление Человека
            if event.type == pygame.KEYDOWN and not game_human.game_over:
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

        # Гравитация Тетриса
        if frame_counter % FALL_SPEED == 0:
            if not game_human.game_over:
                game_human.drop()
            if not game_ailyn.game_over:
                game_ailyn.drop()

        draw_board(screen, game_human, offset_x=50, offset_y=50, title="МАКСУТБЕК (Человек)")
        draw_board(screen, game_ailyn, offset_x=BLOCK_SIZE * COLS + 100, offset_y=50, title="АЙЛИН (Нейросеть)")

        if frame_counter % 10 == 0 and not game_ailyn.game_over: #limiter for Ailyn (reaction time)

            full_screen = pygame.surfarray.array3d(screen)
            matrix_form = full_screen.transpose(1, 0, 2)

            FOV_SIZE = 400

            #focus_x = BLOCK_SIZE * COLS + 100 + (BLOCK_SIZE * COLS) // 2 #motor cortex
            #focus_y = 50 + (BLOCK_SIZE * ROWS) // 2

            half_fov = FOV_SIZE // 2

            # Напоминаю: стакан Айлин по Y от 50 до 650, по X от 400 до 700.
            ailyn_fov = matrix_form[focus_y - half_fov : focus_y + half_fov, focus_x - half_fov : focus_x + half_fov]

            # 4. Отдаем кусок экрана в Зрительную кору!
            ailyn_eye.process_vision(ailyn_fov)

            # --- Моторика (пока заглушка) ---
            decision = ailyn_make_decision(game_ailyn)
            if decision == 'LEFT':
                game_ailyn.move(-1, 0)
            elif decision == 'RIGHT':
                game_ailyn.move(1, 0)
            elif decision == 'ROTATE':
                game_ailyn.rotate()
            elif decision == 'DROP':
                game_ailyn.move(0, 1)

                # Защита от выхода взгляда за пределы экрана!
                # Ширина экрана 1500, высота 700. Половина FOV = 200.
                # Глаз не может подойти к краю ближе, чем на 200 пикселей.

            #focus_x = max(200, min(1500 - 200, focus_x))
            #focus_y = max(200, min(700 - 200, focus_y))


        pygame.display.flip()
        clock.tick(FPS)


    pygame.quit()


if __name__ == "__main__":
    main()