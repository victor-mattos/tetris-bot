from pyboy.pyboy import *
from pyboy import WindowEvent
from keras.models import Sequential
from keras.layers import Dense, Flatten

class TetrisAgent:
    def __init__(self, model):
        self.model = model

        # Define a mapping from predicted move to WindowEvent
        self.move_to_event = {
            0: WindowEvent.PRESS_ARROW_LEFT,
            1: WindowEvent.PRESS_ARROW_RIGHT,
            2: WindowEvent.PRESS_ARROW_UP,
            3: WindowEvent.PRESS_ARROW_DOWN,
            4: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B
        }

    def predict_move(self, game_state):
        # Assume you have a model that predicts the best move given the game state
        predicted_move = self.model.predict(game_state)
        return predicted_move

    def press_key(self, event):
        # Press the mapped event in the game
        pyboy.send_input(event)

    def play_turn(self, game_state):
        # Predict the move
        predicted_move = self.predict_move(game_state)

        # Get the WindowEvent corresponding to the predicted move
        event_to_press = self.move_to_event[predicted_move]

        # Press the mapped event in the game
        self.press_key(event_to_press)