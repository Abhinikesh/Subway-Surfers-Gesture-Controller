import cv2
import time
import json
import os
import numpy as np
from modules.pose_detector import PoseDetector
from modules.gesture_engine import GestureEngine
from modules.keyboard_controller import KeyboardController
from modules.overlay_ui import OverlayUI, SessionStats, make_instructions_panel
import config

CALIB_FILE = "calibration.json"
WIN_NAME   = "Subway Surfers — Body Control"

def show_splash():
    print("\033[96m")
    print("  ███████╗██╗   ██╗██████╗ ██╗    ██╗ █████╗ ██╗   ██╗")
    print("  ██╔════╝██║   ██║██╔══██╗██║    ██║██╔══██╗╚██╗ ██╔╝")
    print("  ███████╗██║   ██║██████╔╝██║ █╗ ██║███████║ ╚████╔╝ ")
    print("  ╚════██║██║   ██║██╔══██╗██║███╗██║██╔══██║  ╚██╔╝  ")
    print("  ███████║╚██████╔╝██████╔╝╚███╔███╔╝██║  ██║   ██║   ")
    print("  ╚══════╝ ╚═════╝ ╚═════╝  ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ")
    print("       SURFERS — Body Gesture Controller v2.0")
    print("   MediaPipe + OpenCV + pynput  |  macOS M2\033[0m\n")

def calibration_screen(cap, detector):
    steps = [
        ("STAND STRAIGHT",  "Stand naturally — press SPACE"),
        ("RAISE BOTH HANDS","Raise both hands above head — press SPACE"),
        ("RAISE LEFT HAND", "Raise only LEFT hand — press SPACE"),
        ("SQUAT DOWN",      "Bend knees slightly — press SPACE"),
    ]
    saved = {}
    for i, (title, instruction) in enumerate(steps):
        print(f"[CALIB] Step {i+1}/4: {instruction}")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT))

            dark = frame.copy()
            cv2.rectangle(dark, (0,0), (frame.shape[1], frame.shape[0]), (5,10,20), -1)
            cv2.addWeighted(dark, 0.55, frame, 0.45, 0, frame)

            cv2.putText(frame, f"CALIBRATION  {i+1} / 4",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,245,255), 2)
            cv2.putText(frame, title,
                        (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,136), 2)
            cv2.putText(frame, instruction,
                        (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, "Press SPACE to capture  |  Q to skip",
                        (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,111,74), 1)

            result = detector.process(frame)
            if result:
                frame = detector.draw_skeleton(frame, result)
                cv2.putText(frame, "Pose detected!", (20,185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,136), 1)
            else:
                cv2.putText(frame, "No pose — step back from camera",
                            (20,185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,110,255), 1)

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if result:
                    saved[title] = True
                    print(f"[CALIB] Step {i+1} captured OK")
                    break
                else:
                    print("[CALIB] No pose detected — try again")
            elif key == ord('q'):
                print("[CALIB] Skipped")
                break

    with open(CALIB_FILE, "w") as f:
        json.dump(saved, f)
    print("[CALIB] Saved!\n")
    return saved


def main():
    show_splash()

    print("[INIT] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found!")
        print("        Go to: System Settings → Privacy → Camera → enable Terminal")
        input("Press Enter to exit...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
    print("[INIT] Webcam OK")

    print("[INIT] Loading MediaPipe...")
    detector = PoseDetector()
    engine   = GestureEngine(detector)
    keyboard = KeyboardController()
    ui       = OverlayUI(config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT)
    stats    = SessionStats()
    print("[INIT] Ready!\n")

    if os.path.exists(CALIB_FILE):
        print("[INIT] Calibration file found — skipping")
    else:
        calibration_screen(cap, detector)

    instructions = make_instructions_panel(400, config.WEBCAM_HEIGHT)

    print("━"*55)
    print("  Open Chrome → poki.com/en/g/subway-surfers")
    print("  Click the game, then move your body!")
    print()
    print("  GESTURES:")
    print("  Both hands up     → JUMP")
    print("  Left hand up only → MOVE LEFT")
    print("  Right hand up only→ MOVE RIGHT")
    print("  Squat down        → SLIDE")
    print("  Arms crossed X    → SHIELD / revive")
    print()
    print("  Q=Quit  P=Pause  R=Reset stats  C=Recalibrate")
    print("━"*55 + "\n")

    paused      = False
    fps_timer   = time.time()
    fps         = 0.0
    frame_count = 0

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame drop — retrying...")
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT))

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer   = now

        gesture = "IDLE"

        if not paused:
            result = detector.process(frame)
            if result:
                frame  = detector.draw_skeleton(frame, result)
                gesture = engine.update(result)
                if gesture != "IDLE":
                    keyboard.send(gesture)
                    stats.record(gesture)
                    print(f"  → {gesture}")
            else:
                cv2.putText(frame, "NO PERSON DETECTED — step back",
                            (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,110,255), 2)
        else:
            cv2.putText(frame, "PAUSED  (P to resume)",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,245,255), 2)

        conf  = engine.confidence()
        frame = ui.draw(frame, gesture, conf, stats, fps)

        combined = np.hstack([frame, instructions])
        cv2.imshow(WIN_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif key == ord('r'):
            stats = SessionStats()
            print("[INFO] Stats reset")
        elif key == ord('c'):
            if os.path.exists(CALIB_FILE):
                os.remove(CALIB_FILE)
            calibration_screen(cap, detector)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "═"*40)
    print("  SESSION SUMMARY")
    print(f"  Jumps:    {stats.jumps}")
    print(f"  Slides:   {stats.slides}")
    print(f"  Lefts:    {stats.lefts}")
    print(f"  Rights:   {stats.rights}")
    print(f"  Calories: {stats.calories} kcal")
    print(f"  Time:     {stats.elapsed}")
    print("═"*40 + "\n")

if __name__ == "__main__":
    main()
