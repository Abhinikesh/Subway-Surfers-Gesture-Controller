import cv2
import time
from modules.pose_detector import PoseDetector, PoseResult
import config

detector = PoseDetector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Stand in front of camera and try LEFT/RIGHT arm movements")
print("Watch the values printed — press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    result = detector.process(frame)
    if result:
        frame = detector.draw_skeleton(frame, result)
        d = detector

        l_wrist    = d.get_landmark(result, d.LEFT_WRIST)
        r_wrist    = d.get_landmark(result, d.RIGHT_WRIST)
        l_shoulder = d.get_landmark(result, d.LEFT_SHOULDER)
        r_shoulder = d.get_landmark(result, d.RIGHT_SHOULDER)

        if all([l_wrist, r_wrist, l_shoulder, r_shoulder]):
            body_cx    = (l_shoulder.x + r_shoulder.x) / 2
            shoulder_w = abs(r_shoulder.x - l_shoulder.x)
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2

            # Key values for LEFT detection
            l_out_dist = l_shoulder.x - l_wrist.x
            r_out_dist = r_wrist.x - r_shoulder.x
            needed     = shoulder_w * 0.3

            print(f"shoulder_w={shoulder_w:.3f} | "
                  f"L_out={l_out_dist:.3f}(need>{needed:.3f}) | "
                  f"R_out={r_out_dist:.3f}(need>{needed:.3f}) | "
                  f"L_wrist_y={l_wrist.y:.3f} | "
                  f"R_wrist_y={r_wrist.y:.3f} | "
                  f"shoulder_y={shoulder_y:.3f}")

        # Show values on screen too
        cv2.putText(frame, "Extend arm SIDEWAYS and watch terminal",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,245,255), 1)

    cv2.imshow("Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
