import cv2
import os
import numpy as np
import joblib
import mediapipe as mp

# Path to your trained SVM model
MODEL_PATH = os.path.join("..", "models", "svm.pkl")

# Load Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load trained SVM model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found! Run train.py first.")
clf = joblib.load(MODEL_PATH)

# Feature extractor: flatten hand landmarks
def extract_landmarks(hand_landmarks):
    pts = []
    for lm in hand_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts).reshape(1, -1)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=1
    ) as hands:
        print("Starting real-time hand gesture recognition. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            label = "No Hand"
            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    feats = extract_landmarks(hand_lms)
                    pred = clf.predict(feats)[0]
                    prob = np.max(clf.predict_proba(feats))
                    label = f"{pred} ({prob:.2f})"
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Gesture — Real-Time", frame)

            # Quit with Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
