"""
WEBCAM DEMO - Professional Emotion Recognition System
Features: Real-time analysis, mood tracking, session statistics, logging, alerts
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
from datetime import datetime

# ============================================================
# LOAD MODEL
# ============================================================
print("[SYSTEM] Initializing Emotion Recognition Engine...")
model = load_model('../models/best_cnn_model.h5')
print("[SYSTEM] Model loaded successfully.")

# ============================================================
# FACE DETECTOR
# ============================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ============================================================
# CONFIGURATION
# ============================================================
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
colors = {
    'Angry': (0, 0, 255), 'Disgust': (0, 140, 0), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 80, 80),
    'Surprise': (255, 165, 0)
}

# ============================================================
# ANALYSIS VARIABLES
# ============================================================
history = deque(maxlen=5)
emotion_log = []  # Full session log
session_start = time.time()
fps_history = deque(maxlen=30)
frame_count = 0
mood_changes = 0
last_emotion = None
emotion_durations = {e: 0.0 for e in emotions}
current_emotion_start = time.time()

# ============================================================
# CAMERA
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

print("[SYSTEM] Camera initialized.")
print("[SYSTEM] Press 'Q' to quit | 'S' to save session log | 'R' to reset analytics")
print("=" * 60)

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    frame_start = time.time()
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Equalize histogram for better detection
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray_eq, 1.2, 5, minSize=(60, 60))

    current_emotion = None
    current_confidence = 0
    all_probs = None

    for (x, y, w, h) in faces:
        # Extract and preprocess face
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)

        # Predict
        pred = model.predict(roi, verbose=0)[0]
        history.append(pred)
        avg_pred = np.mean(history, axis=0)
        all_probs = avg_pred

        idx = np.argmax(avg_pred)
        current_emotion = emotions[idx]
        current_confidence = avg_pred[idx] * 100
        color = colors[current_emotion]

        # ---- Face bounding box ----
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        # ---- Emotion label ----
        label = f'{current_emotion}  {current_confidence:.1f}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (x, y-th-12), (x+tw+8, y), color, -1)
        cv2.putText(display, label, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ---- Detailed probability panel (right side of face) ----
        panel_x = x + w + 12
        panel_w = 130
        panel_h = 20
        if panel_x + panel_w < display.shape[1]:
            # Panel background
            cv2.rectangle(display, (panel_x-5, y-5), (panel_x+panel_w+40, y+len(emotions)*(panel_h+4)+5), (30, 30, 30), -1)
            cv2.putText(display, 'CLASSIFICATION DETAILS', (panel_x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            for j, (em, prob) in enumerate(zip(emotions, avg_pred)):
                bar_y = y + 28 + j * (panel_h + 3)
                pct = prob * 100

                # Bar background
                cv2.rectangle(display, (panel_x, bar_y), (panel_x + panel_w, bar_y + panel_h), (50, 50, 50), -1)
                # Bar fill
                fill_w = int(panel_w * prob)
                bar_color = colors[em]
                cv2.rectangle(display, (panel_x, bar_y), (panel_x + fill_w, bar_y + panel_h), bar_color, -1)
                # Bar border
                cv2.rectangle(display, (panel_x, bar_y), (panel_x + panel_w, bar_y + panel_h), (100, 100, 100), 1)
                # Percentage text
                cv2.putText(display, f'{em}  {pct:.1f}%', (panel_x + panel_w + 5, bar_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    # ---- Track emotion changes ----
    if current_emotion and current_emotion != last_emotion:
        if last_emotion is not None:
            mood_changes += 1
        duration = time.time() - current_emotion_start
        if last_emotion:
            emotion_durations[last_emotion] += duration
        current_emotion_start = time.time()
        last_emotion = current_emotion

    # ---- Log emotion ----
    if current_emotion:
        emotion_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'emotion': current_emotion,
            'confidence': f'{current_confidence:.1f}%'
        })

    # ---- FPS ----
    fps = 1.0 / (time.time() - frame_start + 0.001)
    fps_history.append(fps)
    avg_fps = np.mean(fps_history)

    # ---- Calculate session statistics ----
    session_duration = time.time() - session_start
    if len(emotion_log) > 0:
        from collections import Counter
        emotion_counts = Counter([e['emotion'] for e in emotion_log])
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_pct = (emotion_counts[dominant_emotion] / len(emotion_log)) * 100
    else:
        dominant_emotion = 'N/A'
        dominant_pct = 0

    # ---- HEADER PANEL ----
    cv2.rectangle(display, (0, 0), (800, 70), (20, 20, 20), -1)
    cv2.putText(display, 'EMOTION RECOGNITION SYSTEM', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(display, f'Session: {session_duration:.0f}s | FPS: {avg_fps:.0f} | Faces: {len(faces)} | Frame: {frame_count}',
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(display, f'Dominant: {dominant_emotion} ({dominant_pct:.0f}%) | Mood Changes: {mood_changes}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # ---- FOOTER PANEL (Emotion distribution bar) ----
    footer_y = 580
    cv2.rectangle(display, (0, footer_y), (800, 600), (20, 20, 20), -1)
    if len(emotion_log) > 0:
        bar_width = 114  # 800 / 7
        for i, em in enumerate(emotions):
            pct = (emotion_counts.get(em, 0) / len(emotion_log)) * 100
            bar_h = int(pct * 0.2)  # Max 20px height
            bar_x = i * bar_width
            cv2.rectangle(display, (bar_x + 2, footer_y + 18 - bar_h), (bar_x + bar_width - 2, footer_y + 18), colors[em], -1)
            cv2.putText(display, f'{em}', (bar_x + 5, footer_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    # ---- No face detected ----
    if len(faces) == 0:
        cv2.putText(display, 'NO FACE DETECTED', (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        cv2.putText(display, 'Position your face in front of the camera', (260, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    # ---- Show ----
    cv2.imshow('Emotion Recognition System | PFA', display)

    # ---- Keyboard ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save session log
        filename = f'session_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(filename, 'w') as f:
            f.write(f"EMOTION RECOGNITION SESSION LOG\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {session_duration:.0f}s\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Mood changes: {mood_changes}\n")
            f.write(f"Dominant emotion: {dominant_emotion} ({dominant_pct:.0f}%)\n")
            f.write(f"\nEmotion Distribution:\n")
            for em in emotions:
                f.write(f"  {em}: {emotion_counts.get(em, 0)} ({emotion_counts.get(em, 0)/len(emotion_log)*100:.1f}%)\n")
            f.write(f"\nDetailed Log:\n")
            for entry in emotion_log[-100:]:
                f.write(f"  [{entry['timestamp']}] {entry['emotion']} ({entry['confidence']})\n")
        print(f"[SYSTEM] Log saved: {filename}")
    elif key == ord('r'):
        emotion_log.clear()
        emotion_durations = {e: 0.0 for e in emotions}
        mood_changes = 0
        session_start = time.time()
        frame_count = 0
        print("[SYSTEM] Analytics reset.")

# ============================================================
# SHUTDOWN
# ============================================================
print("=" * 60)
print(f"[SYSTEM] Session ended.")
print(f"[SYSTEM] Duration: {time.time()-session_start:.0f}s")
print(f"[SYSTEM] Frames processed: {frame_count}")
print(f"[SYSTEM] Mood changes detected: {mood_changes}")
cap.release()
cv2.destroyAllWindows()