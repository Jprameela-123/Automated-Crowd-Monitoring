# Automated Monitoring and Recognition of Unusual Crowd Activities for Public Safety Applications.

## Overview
The key idea of this work is to identify the abnormal crowd activities in public crowded areas.Generally the surveillance systems capture the video but it cann't detect the events.By this work we implement a feature that  is surveillance systems itself detects the abnormal crowd activities in public areas through advanced deeplearning techniques without human intervention.
The application analyzes uploaded crowd videos, classifies crowd behavior as either **Normal** or **Abnormal**, displays annotated video frames, and automatically sends an email notification whenever an abnormal event is detected.

---

## Objectives
Crowd Anomaly Detection Using CNN, GRU, and Fuzzy Logic
- Detect abnormal crowd behavior from surveillance videos.
- Extract spatial features using CNN.
- Learn temporal motion patterns using GRU.
- Fuse CNN and GRU predictions using Fuzzy Logic.
- Display annotated video output.
- Generate automatic email alerts upon anomaly detection.

---

## Project Directory Structure

```
CrowdAnomalyDetection/
│
├── app.py
├── pipeline.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── cnn_model.h5
│   ├── model.h5
│   └── fuzzy_model.pth
│
├── training/
│   ├── cnn_training.ipynb
│   ├── gru_training.ipynb
│   └── fusion-fuzzy_training.py
│
├── ── testing/
│   ├── sample_videos/
│   │   └── Fighting006_x264.mp4
│   │
│   └── evaluation/
│       └── evaluate.py
│
outputs/
    ├── annotated_video.mp4
    ├── email alerts/
    └── evaluation_results/
│
└── dataset/
│   ├── train/
│   │   ├── normal/
│   │   └── abnormal/
│   │
│   └── test/
│       ├── normal/
│       └── abnormal/
```

---
## Technologies Used

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Fuzzy System | PyTorch |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Performance Evaluation | Scikit-learn |

---
## How to run this project
step1-Initially prepare the dataset in the required format.
step2-Train the models on the dataset consists of normal and abnormal crowd activity videos that are collected from different sources and save them.
step3-Load the trained model on local computer and run the pipeline.py code in which functions are defined to allign the outputs comes from CNN and GRU models,to combine the scores of CNN and GRU using fusion-fuzzy logic,to create annotations and email alerts.
step4-Run app.py by using the below steps
## Required Python Packages

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Upload a surveillance video through the web interface. The system processes the video, detects abnormal crowd activities, displays annotated frames, and sends an email alert if an anomaly is detected.

---

step5-Run evaluate.py to test how the model is performed.

## Evaluation

Run the evaluation script using:

```bash
python evaluate.py
```

The evaluation module computes the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics are generated separately for:

- CNN Model
- GRU Model
- CNN-GRU-FUZZY fusion Model

---

## License
This project has been developed for academic purpose.
