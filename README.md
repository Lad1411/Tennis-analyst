# 🎾 Tennis Analyst

An automated computer vision tool for analyzing tennis match footage. This project leverages state-of-the-art deep learning models to detect players and track the tennis ball in court videos, enabling advanced sports analytics.

## 🌟 Features

* **Player Detection:** Utilizes **YOLOv5su** (Ultralytics) for robust and real-time detection of tennis players on the court.
* **Ball Tracking:** Implements **TrackNet v1** to accurately track the high-speed, often blurred tennis ball across frames.
* **Match Analytics:** (Add specific analytics your code does, e.g., rally count, ball speed estimation, player heatmaps, bounce detection).
* **Video Annotation:** Outputs an annotated video with bounding boxes for players and trajectory lines for the ball.

## 🏗️ Model Architecture

### YOLOv5su

We use the updated `yolov5su` model from Ultralytics. It provides an excellent balance between inference speed and accuracy, making it ideal for tracking human figures in sports broadcasts.

### TrackNet v1

Tennis balls are notoriously difficult to track due to motion blur, occlusion, and small size. TrackNet v1 addresses this by taking consecutive frames as input to generate a heat map indicating the ball's location, effectively predicting the trajectory even when the ball is temporarily invisible.

## 📋 Prerequisites

Ensure you have the following installed before running the project:
* Python 3.8 or higher
* [PyTorch](https://pytorch.org/) (compiled with CUDA for GPU acceleration, highly recommended)
* OpenCV

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lad1411/Tennis-analyst.git
   cd Tennis-analyst

2. **Create a virtual environment (optional but recommended):**

```bash
    python -m venv venv
    venv\Scripts\activate
    
```

