# Multimodal Stress Detection (rPPG + FER)

This project uses a webcam to detect stress levels in real-time by combining **Remote Photoplethysmography (rPPG)** signals with **Facial Expression Recognition (FER)**.

It estimates heart rate (BPM) and Heart Rate Variability (HRV) purely from video, fusing these metrics with emotion detection to classify your state as *Relaxed*, *Stressed*, *Excited*, or *Fatigued*.

## Features

- **Non-contact Vitals**: Extracts BPM and HRV (SDNN, RMSSD) from facial video.
- **Emotion Recognition**: Uses `HSEmotion` (EfficientNet) to detect facial expressions.
- **Fusion Engine**: Combines physiological signals and facial cues for robust state classification.
- **Data Logging**: Records session metrics to CSV for post-hoc analysis.
- **Privacy-First**: All processing runs locally on your machine.

## Installation

Requires Python 3.10+ (recommended).

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MatteoZak/rPPG.git
    cd rppg_project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Start the detection system:

```bash
python main.py
```

### Controls
- `q` : Quit application
- `r` : Reset tracking/baseline
- `l` : Toggle data logging (REC indicator will appear)

## Analysis

If you recorded a session (using 'l' to toggle logging), you can visualize the results:

```bash
# Analyze the latest session
python analyze_session.py

# Analyze a specific file
python analyze_session.py logs/session_20240210_120000.csv
```

This generates:
- `_timeline.png`: Graphs of BPM, SQI, and Stress over time.
- `_summary.png`: Pie charts of dominant states and emotions.

## How it works

The system uses a two-stream approach:
1.  **rPPG Stream**: Extracts the subtle color changes in skin caused by blood flow (the pulse signal).
2.  **FER Stream**: Analyzes facial landmarks to determine emotional valence.

The **Fusion Engine** applies logic to these inputs. For example, high heart rate + "Fear" expression might indicate *High Stress*, while high heart rate + "Happy" might indicate *Excitement*.
