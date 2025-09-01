import cv2
import os
import csv
import time
import numpy as np
import mediapipe as mp

DATA_DIR = os.path.join("..", "data")
CSV_PATH = os.path.join(DATA_DIR, "features.csv")
os.makedirs(DATA_DIR, exist_ok=True)

GESTURE_LABEL = os.environ.get("GESTURE_LABEL", "palm")  # set via env or edit here
SAMPLES_PER_CLASS = int(os.environ.get("SAMPLES", "400"))  # ~400 frames per gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(hand_landmarks):
    # 21 landmarks, each with (x, y, z)
    pts = []
    for lm in hand_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return pts  # length 63

def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        header = [f"f{i}" for i in range(63)] + ["label"]
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def main():
    ensure_csv_header()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    collected = 0
    print(f"[INFO] Collecting {SAMPLES_PER_CLASS} samples for label: {GESTURE_LABEL}")
    print("[INFO] Starting in 3 seconds...")
    time.sleep(3)

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1
    ) as hands:

        while collected < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret: break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    feats = extract_landmarks(hand_lms)
                    with open(CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(feats + [GESTURE_LABEL])
                    collected += 1

                    # draw for user feedback
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(frame, f"{GESTURE_LABEL}: {collected}/{SAMPLES_PER_CLASS}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collecting", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
