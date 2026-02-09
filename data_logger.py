import csv
import os
import time
from datetime import datetime


class DataLogger:

    def __init__(self, output_dir="logs", prefix="session"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.file = None
        self.writer = None
        self.start_time = None
        self.filepath = None
        self.row_count = 0

        self.columns = [
            'timestamp', 'elapsed', 'bpm', 'sqi',
            'rmssd', 'sdnn', 'pnn50', 'lf_hf', 'breathing_rate',
            'emotion', 'emotion_confidence', 'hrv_stress_score', 'state'
        ]

    def start(self):
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{timestamp}.csv"
        self.filepath = os.path.join(self.output_dir, filename)

        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        self.file.flush()

        self.start_time = time.time()
        self.row_count = 0

        print(f"[Logger] Started: {self.filepath}")

    def log(self, bpm, sqi, hrv_metrics, emotion, emotion_confidence,
            hrv_stress_score, state):
        if self.writer is None:
            return

        now = time.time()
        elapsed = now - self.start_time

        row = {
            'timestamp': f"{now:.3f}",
            'elapsed': f"{elapsed:.2f}",
            'bpm': f"{bpm:.1f}" if bpm > 0 else "",
            'sqi': f"{sqi:.3f}" if sqi > 0 else "",
            'rmssd': f"{hrv_metrics.get('rmssd', 0):.1f}" if hrv_metrics.get('rmssd', 0) > 0 else "",
            'sdnn': f"{hrv_metrics.get('sdnn', 0):.1f}" if hrv_metrics.get('sdnn', 0) > 0 else "",
            'pnn50': f"{hrv_metrics.get('pnn50', 0):.1f}" if hrv_metrics.get('pnn50', 0) > 0 else "",
            'lf_hf': f"{hrv_metrics.get('lf_hf', 0):.2f}" if hrv_metrics.get('lf_hf', 0) > 0 else "",
            'breathing_rate': f"{hrv_metrics.get('breathing_rate', 0):.1f}" if hrv_metrics.get('breathing_rate', 0) > 0 else "",
            'emotion': emotion,
            'emotion_confidence': f"{emotion_confidence:.1f}" if emotion_confidence > 0 else "",
            'hrv_stress_score': f"{hrv_stress_score:.3f}" if hrv_stress_score is not None else "",
            'state': state
        }

        self.writer.writerow(row)
        self.row_count += 1

        if self.row_count % 30 == 0:
            self.file.flush()

    def stop(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

            duration = time.time() - self.start_time if self.start_time else 0
            print(f"[Logger] Stopped. {self.row_count} rows in {duration:.1f}s")
            print(f"[Logger] Saved: {self.filepath}")

            return self.filepath
        return None

    def is_logging(self):
        return self.file is not None

    def get_filepath(self):
        return self.filepath
