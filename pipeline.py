
import cv2
import numpy as np
import tensorflow as tf
import smtplib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from email.mime.text import MIMEText
import socket
SEQ_LEN = 10
# ---------------- EMAIL FUNCTION ----------------
def send_email_alert(message):
    sender = "abc@gmail.com"
    receiver = "abc@gmail.com"
    password = "rksm"  
    msg = MIMEText(message)
    msg["Subject"] = "🚨 Crowd Anomaly Alert"
    msg["From"] = sender
    msg["To"] = receiver
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("Email Sent")
    except Exception as e:
        print("Email Failed:", e)
# ---------------- GET IP ADDRESS ----------------
def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(
            hostname
        )
        return ip_address
    except Exception:
        return "Unknown"
# ---------------- OPTICAL FLOW ----------------
def compute_flow(frames):
    flows = []
    prev = cv2.cvtColor(
        (frames[0] * 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    )
    for i in range(1, len(frames)):
        curr = cv2.cvtColor(
            (frames[i] * 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        )
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )
        mag, ang = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )
        mag = mag / (np.max(mag) + 1e-6)
        ang = ang / (np.max(ang) + 1e-6)

        flows.append(
            np.stack([mag, ang], axis=-1)
        )
        prev = curr
    return np.array(flows)

# ---------------- SEQUENCES ----------------
def create_sequences(flow):
    X = []
    for i in range(len(flow) - SEQ_LEN):
        X.append(
            flow[i:i + SEQ_LEN]
        )
    return np.array(X)

# ---------------- CNN ----------------
def cnn_predict(frames, model):
    return model.predict(
        np.array(frames),
        verbose=0
    ).flatten()

# ---------------- GRU ----------------
def gru_predict(flow, model):
    seq = create_sequences(flow)
    return model.predict(
        seq,
        verbose=0
    ).flatten()

# ---------------- ALIGN ----------------
def align(a, b):
    m = min(len(a), len(b))
    return a[:m], b[:m]

# ---------------- NORMALIZE ----------------
def normalize(x):
    x = np.array(x)
    if np.max(x) - np.min(x) == 0:
        return x
    return (
        x - np.min(x)
    ) / (
        np.max(x) - np.min(x) + 1e-6
    )

# ---------------- FUSION ----------------
def fusion(cnn, gru):
    cnn, gru = align(cnn, gru)

    cnn = normalize(cnn)
    gru = normalize(gru)

    return 0.6 * cnn + 0.4 * gru

# ---------------- FUZZY LOGIC ----------------
fused_input = ctrl.Antecedent(
    np.arange(0, 1.01, 0.01),
    'fused'
)
anomaly = ctrl.Consequent(
    np.arange(0, 1.01, 0.01),
    'anomaly'
)
# Membership Functions
fused_input['low'] = fuzz.trimf(
    fused_input.universe,
    [0, 0, 0.5]
)
fused_input['medium'] = fuzz.trimf(
    fused_input.universe,
    [0.2, 0.5, 0.8]
)
fused_input['high'] = fuzz.trimf(
    fused_input.universe,
    [0.5, 1, 1]
)
anomaly['normal'] = fuzz.trimf(
    anomaly.universe,
    [0, 0, 0.4]
)
anomaly['suspicious'] = fuzz.trimf(
    anomaly.universe,
    [0.3, 0.5, 0.7]
)
anomaly['abnormal'] = fuzz.trimf(
    anomaly.universe,
    [0.6, 1, 1]
)
# Rules
rule1 = ctrl.Rule(
    fused_input['low'],
    anomaly['normal']
)
rule2 = ctrl.Rule(
    fused_input['medium'],
    anomaly['suspicious']
)
rule3 = ctrl.Rule(
    fused_input['high'],
    anomaly['abnormal']
)
# Control System
anomaly_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3]
)
anomaly_sim = ctrl.ControlSystemSimulation(
    anomaly_ctrl
)
# ---------------- FUZZY DETECTION ----------------
def fuzzy_detection(fused_scores):
    results = []
    for score in fused_scores:
        anomaly_sim.input['fused'] = float(score)
        anomaly_sim.compute()
        results.append(
            anomaly_sim.output['anomaly']
        )
    return np.array(results)

# ---------------- DETECTION ----------------
def detect_event(scores):
    scores = normalize(scores)
    threshold = (
        np.mean(scores)
        + 0.4 * np.std(scores)
    )
    return (
        scores > threshold
    ).astype(int)

# ---------------- PIPELINE ----------------
def run_pipeline(frames, cnn_model, gru_model):
    cnn_scores = cnn_predict(
        frames,
        cnn_model
    )
    flow = compute_flow(frames)
    gru_scores = gru_predict(
        flow,
        gru_model
    )
    cnn_scores, gru_scores = align(
        cnn_scores,
        gru_scores
    )
    # Weighted Fusion
    fused = fusion(
        cnn_scores,
        gru_scores
    )
    # Fuzzy Logic
    fuzzy_scores = fuzzy_detection(
        fused
    )
    # Final Detection
    states = detect_event(
        fuzzy_scores
    )
    start = (
        np.where(states == 1)[0][0]
        if np.any(states == 1)
        else None
    )
    end = (
        np.where(states == 1)[0][-1]
        if np.any(states == 1)
        else None
    )
    return {
        "cnn_scores": cnn_scores,
        "gru_scores": gru_scores,
        "fusion": fused,
        "fusion_scores": fuzzy_scores,
        "states": states,
        "start": start,
        "end": end
    }
# ---------------- ANNOTATION ----------------
def annotate(frames, states):
    out = []
    prev = (
        frames[0] * 255
    ).astype(np.uint8)
    for i in range(len(frames)):
        frame = (
            frames[i] * 255
        ).astype(np.uint8)
        state = (
            states[i]
            if i < len(states)
            else 0
        )
        if state == 1:
            bbox = get_bbox(
                frame,
                prev
            )
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    frame,
                    "ANOMALY",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        prev = frame.copy()
        out.append(frame)
    return out

# ---------------- MOTION BOX ----------------
def get_bbox(frame, prev):
    g1 = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    g2 = cv2.cvtColor(
        prev,
        cv2.COLOR_BGR2GRAY
    )
    flow = cv2.calcOpticalFlowFarneback(
        g2,
        g1,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )
    mag, _ = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )
    mag = (
        mag * 255
    ).astype(np.uint8)
    _, th = cv2.threshold(
        mag,
        25,
        255,
        cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    c = max(
        contours,
        key=cv2.contourArea
    )
    if cv2.contourArea(c) < 200:
        return None
    return cv2.boundingRect(c)













