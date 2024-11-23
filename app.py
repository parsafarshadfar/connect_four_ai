import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time

# Set the page configuration
st.set_page_config(page_title='Connect Four', page_icon='🎮')

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

# Initialize game state
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    st.session_state.game_over = False
    st.session_state.current_player = 1  # 1 for player1/human, 2 for player2/AI
    st.session_state.winner = None
    st.session_state.game_mode = 'One Player'
    st.session_state.player1_name = 'Player 1'
    st.session_state.player2_name = 'AI'

# Ensure 'difficulty' is initialized
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 'Medium'  # Default difficulty

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def reset_game():
    st.session_state.board = create_board()
    st.session_state.game_over = False
    st.session_state.current_player = 1
    st.session_state.winner = None

def is_valid_location(board, col):
    return board[0][col] == 0

def get_valid_locations(board):
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Check horizontal locations
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all([board[r][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Check vertical locations
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all([board[r + i][c] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all([board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all([board[r - i][c + i] == piece for i in range(WINDOW_LENGTH)]):
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    # Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonals
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, float('inf'))
            elif winning_move(board, PLAYER_PIECE):
                return (None, float('-inf'))
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = float('-inf')
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = float('inf')
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def draw_board(board):
    board_flipped = np.flip(board, 0)
    fig, ax = plt.subplots(figsize=(5, 4))  # Decreased figure size
    ax.set_aspect('equal')
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            color = 'white'
            if board_flipped[r][c] == PLAYER_PIECE:
                color = 'red'
            elif board_flipped[r][c] == AI_PIECE:
                color = 'yellow'
            circle = patches.Circle((c + 0.5, r + 0.5), 0.4, color=color, ec='black')
            ax.add_patch(circle)
    plt.xlim(0, COLUMN_COUNT)
    plt.ylim(0, ROW_COUNT)
    plt.axis('off')
    st.pyplot(fig)

def ai_move():
    board = st.session_state.board
    depth = {'Easy': 2, 'Medium': 4, 'Hard': 6}[st.session_state.difficulty]
    col, minimax_score = minimax(board, depth, float('-inf'), float('inf'), True)
    if col is not None and is_valid_location(board, col):
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, AI_PIECE)
        if winning_move(board, AI_PIECE):
            st.session_state.game_over = True
            st.session_state.winner = st.session_state.player2_name
        else:
            st.session_state.current_player = 1

# Sidebar for game settings
st.sidebar.title("Game Settings")
if st.sidebar.button("Reset Game"):
    reset_game()
st.session_state.game_mode = st.sidebar.selectbox("Game Mode", ['One Player', 'Two Players'])

if st.session_state.game_mode == 'One Player':
    st.session_state.player1_name = st.sidebar.text_input("Player Name", value='Player 1')
    st.session_state.player2_name = 'AI'
    st.session_state.difficulty = st.sidebar.select_slider(
        "Difficulty Level",
        options=['Easy', 'Medium', 'Hard'],
        value=st.session_state.difficulty
    )
else:
    st.session_state.player1_name = st.sidebar.text_input("Player 1 Name", value='Player 1')
    st.session_state.player2_name = st.sidebar.text_input("Player 2 Name", value='Player 2')

st.title("Connect Four")
if st.session_state.game_mode == 'One Player':
    st.write(f"Play against the computer! Difficulty: **{st.session_state.difficulty}**")
else:
    st.write("Play with a friend!")

if st.session_state.game_over:
    st.write(f"**{st.session_state.winner} Won the game!**")
    # Show balloons only if the winner is not the AI
    if st.session_state.winner != 'AI':
        st.balloons()
else:
    current_player_name = st.session_state.player1_name if st.session_state.current_player == 1 else st.session_state.player2_name
    if st.session_state.game_mode != 'One Player':
        st.write(f":{'red' if st.session_state.current_player == 1 else 'orange'}[**{current_player_name}'s Turn**]")
    if st.session_state.game_mode == 'One Player' and st.session_state.current_player == 2:
        st.write("**AI is thinking...**")
        time.sleep(0.02)  # Delay to simulate thinking
        ai_move()
        st.rerun()
    else:
        # Center the buttons
        button_container = st.container()
        with button_container:
            # Calculate the required spacing to center the buttons
            total_columns = 14  # Total columns including spacers
            spacer_width = 0.1  # Adjust spacer width as needed
            col_widths = [spacer_width] * ((total_columns - COLUMN_COUNT) // 2) + [1] * COLUMN_COUNT + [spacer_width] * ((total_columns - COLUMN_COUNT) // 2)
            columns = st.columns(col_widths)
            for idx, i in enumerate(range(len(columns))):
                if idx < (total_columns - COLUMN_COUNT) // 2 or idx >= ((total_columns + COLUMN_COUNT) // 2):
                    continue  # Skip spacer columns
                with columns[idx]:
                    col_num = idx - ((total_columns - COLUMN_COUNT) // 2)
                    if st.button("▼", key=f'col_{col_num}_{st.session_state.current_player}'):
                        if is_valid_location(st.session_state.board, col_num):
                            row = get_next_open_row(st.session_state.board, col_num)
                            piece = PLAYER_PIECE if st.session_state.current_player == 1 else AI_PIECE
                            drop_piece(st.session_state.board, row, col_num, piece)
                            if winning_move(st.session_state.board, piece):
                                st.session_state.game_over = True
                                st.session_state.winner = current_player_name
                            else:
                                st.session_state.current_player = 2 if st.session_state.current_player == 1 else 1
                            st.rerun()


draw_board(st.session_state.board)
