from collections import deque
import cv2
import pyautogui
import mss
import mss.tools
import torch
import numpy as np
import time
import easyocr


class Environment:
    def __init__(self):
        self.coins = 0
        self.frames = deque(maxlen=4)
        self.score_region = {"top": 27, "left": 1740, "width": 165 , "height": 50}
        self.coins_region = {"top": 100, "left": 1775, "width": 92 , "height": 50}
        self.gameplay_region = {"top": 240, "left": 350, "width": 1220 , "height": 840}
        self.last_score = None
        self.stall_count = 0
        self.stall_threshold = 10
        self.game_state = "MENU"
        self.play_button_template = cv2.imread("imgs/play.png", cv2.IMREAD_GRAYSCALE)
        self.reader = easyocr.Reader(['en'], gpu=True)

    def reset(self):
        self.coins = 0
        self.last_score = None
        self.stall_count = 0
        self.frames.clear()
        # ------SKIP MENU----------
        while not self.is_play_button():
            pyautogui.press('enter')
            time.sleep(0.2)
        if self.is_play_button():
            pyautogui.press('enter')
            time.sleep(1.5)
            self.game_state = "GAME"
            print('----STARTING GAME----')
            for _ in range(4):
                self.get_gameplay_screen()
            return self.frames_to_tensor()


    def is_play_button(self):
        region = {"top": 950, "left": 1000, "width": 250 , "height": 70}
        area = self.crop_screen(self.grab_screen(), region)
        area = cv2.cvtColor(area, cv2.COLOR_BGRA2GRAY)
        similarity = cv2.matchTemplate(area, self.play_button_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(similarity)
        if max_val > 0.97:
            return True
        else:
            return False

    def step(self, action):
        self.do_action(action)
        self.get_gameplay_screen()

        coins_delta = self.calculate_delta()
        done = self.is_done()
        reward = coins_delta * 0.5
        if done:
            reward -= 10
            time.sleep(1.5)
            self.game_state = "MENU"
        else:
            reward += 0.5
        observation = self.frames_to_tensor()
        return observation, reward, done

    def grab_screen(self):
        monitor = {"top": 0, "left": 0, "width": 1920 , "height": 1080}
        with mss.MSS() as sct:
            screenshot = sct.grab(monitor)
            return np.array(screenshot)

    def crop_screen(self, frame, r):
        return frame[r['top']: r['top'] + r['height'], r['left']: r['left'] + r['width']]

    def get_gameplay_screen(self, window_width=84, window_height=84):
        gameplay = self.crop_screen(self.grab_screen(), self.gameplay_region)
        gameplay = cv2.cvtColor(gameplay, cv2.COLOR_BGRA2GRAY)
        gameplay = cv2.resize(gameplay, (window_width, window_height))
        self.frames.append(gameplay)

    def frames_to_tensor(self):
        state = np.array(self.frames, dtype=np.float32) / 255.0
        return torch.tensor(state, dtype=torch.float32)

    def _scan_coins(self):
        crop = self.crop_screen(self.grab_screen(), self.coins_region)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        result = self.reader.readtext(crop, allowlist='0123456789', detail=0)
        if not result:
            return None
        text = result[0]
        try:
            return int(text.strip())
        except (ValueError, AttributeError):
            return None

    def calculate_delta(self):
        coins_now = self._scan_coins()
        max_delta = 10
        print(f"OCR read: {coins_now}, previous: {self.coins}")
        if coins_now is not None and coins_now > self.coins:
            delta = coins_now - self.coins
            if delta > max_delta:
                delta = 0
            else:
                self.coins = coins_now
        else:
            delta = 0
        return delta

    def is_done(self):
        crop = self.crop_screen(self.grab_screen(), self.score_region)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        result = self.reader.readtext(crop, allowlist='0123456789', detail=0)
        if not result:
            return False
        try:
            current = result
        except ValueError:
            return False

        if self.last_score is None:
            self.last_score = current
            return False
        if current == self.last_score:
            self.stall_count += 1
        else:
            self.stall_count = 0
        self.last_score = current

        if self.stall_count >= self.stall_threshold:
            print("GAME OVER")
            return True
        return False

    def do_action(self, choice):
        action_table = {
            'UP' : 'up',
            'DOWN' : 'down',
            'LEFT' : 'left',
            'RIGHT' : 'right',
            'NONE' : None
        }
        key = action_table[choice]
        if key is not None:
            return pyautogui.press(key)

