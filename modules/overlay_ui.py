import cv2
import numpy as np
import time
from dataclasses import dataclass, field

GESTURE_COLORS = {
    "JUMP":   (0, 245, 255),
    "SLIDE":  (0, 110, 255),
    "LEFT":   (0, 255, 136),
    "RIGHT":  (0, 255, 136),
    "SHIELD": (179, 0, 255),
    "IDLE":   (74, 111, 100),
}

@dataclass
class SessionStats:
    jumps:   int = 0
    slides:  int = 0
    lefts:   int = 0
    rights:  int = 0
    shields: int = 0
    start_time: float = field(default_factory=time.time)

    def record(self, gesture: str):
        if gesture == "JUMP":     self.jumps   += 1
        elif gesture == "SLIDE":  self.slides  += 1
        elif gesture == "LEFT":   self.lefts   += 1
        elif gesture == "RIGHT":  self.rights  += 1
        elif gesture == "SHIELD": self.shields += 1

    @property
    def calories(self) -> float:
        return round(self.jumps*0.3 + self.slides*0.2 +
                     (self.lefts+self.rights)*0.1, 1)

    @property
    def elapsed(self) -> str:
        s = int(time.time() - self.start_time)
        return f"{s//60:02d}:{s%60:02d}"


class OverlayUI:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height

    def _dark_bar(self, frame, y, h, alpha=0.65):
        roi = frame[y:y+h, 0:self.w]
        black = np.zeros_like(roi)
        cv2.addWeighted(black, alpha, roi, 1-alpha, 0, roi)
        frame[y:y+h, 0:self.w] = roi

    def draw(self, frame, gesture, confidence, stats, fps):
        color = GESTURE_COLORS.get(gesture, (74,111,100))

        # Top bar
        self._dark_bar(frame, 0, 52)
        cv2.putText(frame, "SUBWAY SURFERS",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0,245,255), 2)
        label = gesture if gesture != "IDLE" else "TRACKING..."
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        cv2.putText(frame, label, ((self.w-tw)//2, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"FPS {int(fps)}", (self.w-80, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (74,111,100), 1)

        # Bottom bar
        self._dark_bar(frame, self.h-105, 105)

        # Confidence bar
        bx, by, bw = 12, self.h-92, 200
        cv2.putText(frame, f"Confidence: {int(confidence*100)}%",
                    (bx, by-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255,255,255), 1)
        cv2.rectangle(frame, (bx,by), (bx+bw,by+8), (20,30,50), -1)
        fill = int(bw * confidence)
        if fill > 0:
            cv2.rectangle(frame, (bx,by), (bx+fill,by+8), color, -1)

        # Stats
        cv2.putText(frame, f"Jumps:{stats.jumps}",
                    (12, self.h-65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,255,136), 1)
        cv2.putText(frame, f"Slides:{stats.slides}",
                    (130, self.h-65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,255,136), 1)
        cv2.putText(frame, f"Left:{stats.lefts}",
                    (260, self.h-65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,255,136), 1)
        cv2.putText(frame, f"Right:{stats.rights}",
                    (370, self.h-65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,255,136), 1)
        cv2.putText(frame, f"Cal: {stats.calories} kcal",
                    (12, self.h-38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,200,255), 1)
        cv2.putText(frame, f"Time: {stats.elapsed}",
                    (200, self.h-38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0,200,255), 1)

        # Gesture hints bottom
        hints = [
            ("Both UP=JUMP",     "JUMP"),
            ("Left side=LEFT",   "LEFT"),
            ("Right side=RIGHT", "RIGHT"),
            ("Hands join=SLIDE", "SLIDE"),
        ]
        for i,(txt,g) in enumerate(hints):
            c = GESTURE_COLORS[g] if gesture==g else (40,50,70)
            cv2.putText(frame, txt, (12+i*155, self.h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1)

        return frame


def make_instructions_panel(w=400, h=480):
    panel = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, 30):
        cv2.line(panel, (0,y), (w,y), (10,20,35), 1)
    for x in range(0, w, 30):
        cv2.line(panel, (x,0), (x,h), (10,20,35), 1)

    cv2.rectangle(panel, (0,0), (w,2), (0,245,255), -1)

    def t(text, y, scale=0.52, color=(255,255,255), bold=1):
        cv2.putText(panel, text, (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold)

    t("SUBWAY SURFERS",          45, 0.9,  (0,245,255), 2)
    t("Body Gesture Controller", 72, 0.42, (74,111,100))
    cv2.line(panel, (20,84), (w-20,84), (0,245,255), 1)

    t("HOW TO PLAY:",                    108, 0.55, (0,245,255))
    t("1. Open Chrome browser",          135, 0.5)
    t("2. Go to:",                       160, 0.5)
    t("poki.com/en/g/subway-surfers",    185, 0.48, (0,200,255))
    t("3. Click the game to focus it",   210, 0.5)
    t("4. Move your body to play!",      235, 0.5)

    cv2.line(panel, (20,252), (w-20,252), (0,245,255), 1)
    t("GESTURES:", 278, 0.58, (0,245,255))

    rows = [
        ("Both hands UP",        "= JUMP",   (0,245,255)),
        ("Left arm out (side)",  "= LEFT",   (0,255,136)),
        ("Right arm out (side)", "= RIGHT",  (0,255,136)),
        ("Hands join stomach",   "= SLIDE",  (0,110,255)),
        ("Cross arms chest X",   "= SHIELD", (179,0,255)),
    ]
    for i,(move,action,col) in enumerate(rows):
        y = 308 + i*32
        cv2.putText(panel, move,   (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
        cv2.putText(panel, action, (240,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

    cv2.line(panel, (20,462), (w-20,462), (30,40,60), 1)
    t("Q=Quit  C=Recalibrate  P=Pause  R=Reset",
      478, 0.36, (74,111,100))

    return panel
