import time
from pynput.keyboard import Key, Controller

GESTURE_KEY_MAP = {
    "JUMP":   Key.up,
    "SLIDE":  Key.down,
    "LEFT":   Key.left,
    "RIGHT":  Key.right,
    "SHIELD": Key.space,
}

class KeyboardController:
    def __init__(self):
        self.keyboard = Controller()

    def send(self, gesture: str):
        key = GESTURE_KEY_MAP.get(gesture)
        if key:
            self.keyboard.press(key)
            time.sleep(0.05)
            self.keyboard.release(key)
