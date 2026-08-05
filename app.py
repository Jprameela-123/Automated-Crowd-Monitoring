
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import socket

from pipeline import (
    run_pipeline,
    annotate,
    send_email_alert
)

# ---------------- LOAD MODELS ----------------
cnn_model = tf.keras.models.load_model("cnn_model.h5")
gru_model = tf.keras.models.load_model("gru_model.h5")

# ---------------- TITLE ----------------
st.title("🚨 Crowd Anomaly Detection System")

# ---------------- VIDEO UPLOAD ----------------
video = st.file_uploader("Upload Video", type=["mp4"])

# ---------------- GET DEVICE IP ----------------
def get_device_ip():
    return socket.gethostbyname(socket.gethostname())

# ---------------- PROCESS ----------------
if video:

    with open("temp.mp4", "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture("temp.mp4")
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # ---------------- PIPELINE ----------------
    result = run_pipeline(frames, cnn_model, gru_model)

    np.save("cnn_scores.npy", result["cnn_scores"])
    np.save("gru_scores.npy", result["gru_scores"])
    np.save("fusion_scores.npy", result["fusion_scores"])
    np.save("y_true.npy", result["states"])

    annotated = annotate(frames, result["states"])

    st.subheader("🎥 Output Video")

    placeholder = st.empty()

    for frame in annotated:
        placeholder.image(frame, channels="BGR")
        time.sleep(1 / 25)

    # ---------------- FINAL ANALYSIS ----------------
    device_ip = get_device_ip()

    if result["start"] is not None:

        fps = 25

        start_time = round(result["start"] / fps, 2)
        end_time = round(result["end"] / fps, 2)

        msg = f"""
🚨 CROWD ANOMALY DETECTED

Start Frame: {result['start']}
End Frame: {result['end']}

Start Time: {start_time}s
End Time: {end_time}s

Device IP Address: {device_ip}
"""

        send_email_alert(msg)

        final = annotated[-1].copy()
        h, w = final.shape[:2]

        cv2.rectangle(final, (0, 0), (w, h), (0, 0, 255), 5)
        cv2.putText(final, "FINAL: ABNORMAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.error("🚨 Anomaly Detected")

    else:

        final = annotated[-1].copy()
        h, w = final.shape[:2]

        cv2.rectangle(final, (0, 0), (w, h), (0, 255, 0), 5)
        cv2.putText(final, "FINAL: NORMAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        st.success("✅ Normal Crowd Detected")

    st.image(final, channels="BGR")
    