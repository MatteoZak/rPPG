import cv2
import numpy as np
from collections import deque
import time

from rppg_processor import RPPGProcessor
from emotion_engine import EmotionEngine
from fusion_engine import FusionEngine
from data_logger import DataLogger


class StressDetectionApp:

    def __init__(self, rppg_model='EfficientPhys.rlap', enable_logging=True):
        self.rppg = RPPGProcessor(model_name=rppg_model, fps=30)
        self.emotion = EmotionEngine(detection_interval=15)
        self.fusion = FusionEngine(bpm_threshold=90)

        self.enable_logging = enable_logging
        self.logger = DataLogger(output_dir="logs") if enable_logging else None
        self.logging_active = False

        self.window_name = "Multimodal Stress Detection"
        self.graph_width = 400
        self.graph_height = 150

        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

    def draw_ui_overlay(self, frame, bpm, emotion, emotion_confidence, state,
                        state_color, fps, sqi, hrv_metrics):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        if bpm > 0:
            sqi_bars = int(sqi * 5)
            sqi_color = (0, 255, 0) if sqi > 0.7 else (0, 255, 255) if sqi > 0.4 else (0, 0, 255)

            bpm_text = f"BPM: {int(bpm)} "
            cv2.putText(frame, bpm_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            bar_x = 150
            for i in range(5):
                color = sqi_color if i < sqi_bars else (100, 100, 100)
                cv2.rectangle(frame, (bar_x + i*10, 18), (bar_x + i*10 + 8, 28), color, -1)

            sqi_text = f" SQI:{sqi:.2f}"
            cv2.putText(frame, sqi_text, (bar_x + 55, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "BPM: -- (acquiring...)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        rmssd = hrv_metrics.get('rmssd', 0)
        sdnn = hrv_metrics.get('sdnn', 0)
        pnn50 = hrv_metrics.get('pnn50', 0)
        lf_hf = hrv_metrics.get('lf_hf', 0)

        if rmssd > 0:
            hrv_text = f"HRV: RMSSD {int(rmssd)}ms  SDNN {int(sdnn)}ms  pNN50 {pnn50:.1f}%"
            hrv_color = (0, 255, 0) if rmssd > 40 else (0, 255, 255) if rmssd > 25 else (0, 100, 255)
        else:
            hrv_text = "HRV: -- (collecting data...)"
            hrv_color = (100, 100, 100)

        cv2.putText(frame, hrv_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hrv_color, 2)

        if lf_hf > 0:
            lf_hf_text = f"LF/HF: {lf_hf:.2f}"
            lf_hf_color = (0, 100, 255) if lf_hf > 2.0 else (0, 255, 0)
            cv2.putText(frame, lf_hf_text, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, lf_hf_color, 2)

        breathing = hrv_metrics.get('breathing_rate', 0)
        if breathing > 0:
            br_text = f"Breathing: {breathing:.1f} rpm"
            cv2.putText(frame, br_text, (200, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        conf_text = f" ({emotion_confidence:.0f}%)" if emotion_confidence > 0 else ""
        emotion_text = f"Emotion: {emotion}{conf_text}"
        cv2.putText(frame, emotion_text, (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        model_text = f"rPPG: {self.rppg.model_name} | FER: HSEmotion ENet-B2"
        cv2.putText(frame, model_text, (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.logging_active:
            cv2.circle(frame, (w - 30, 140), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        state_text = f"STATE: {state}"
        text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = w - text_size[0] - 20
        text_y = 60

        cv2.rectangle(frame,
                      (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      state_color, -1)

        cv2.putText(frame, state_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    def flip_face_box(self, box, frame_width):
        if box is None:
            return None

        flipped = box.copy()
        x1, x2 = int(box[1][0]), int(box[1][1])
        new_x1 = frame_width - x2
        new_x2 = frame_width - x1
        flipped[1][0] = new_x1
        flipped[1][1] = new_x2
        return flipped

    def draw_face_box(self, frame, box):
        if box is None:
            return

        y1, y2 = int(box[0][0]), int(box[0][1])
        x1, x2 = int(box[1][0]), int(box[1][1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Face ROI", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_signal_graph(self, frame):
        h, w = frame.shape[:2]
        filtered_signal = self.rppg.get_filtered_signal()

        if len(filtered_signal) == 0:
            return

        signal_to_plot = filtered_signal[-self.graph_width:]

        if len(signal_to_plot) > 1:
            signal_min = np.min(signal_to_plot)
            signal_max = np.max(signal_to_plot)
            signal_range = signal_max - signal_min

            if signal_range > 0:
                normalized = (signal_to_plot - signal_min) / signal_range
            else:
                normalized = np.zeros_like(signal_to_plot)

            graph_y = h - self.graph_height - 10
            graph_x = w - self.graph_width - 10

            cv2.rectangle(frame,
                          (graph_x, graph_y),
                          (graph_x + self.graph_width, graph_y + self.graph_height),
                          (0, 0, 0), -1)

            cv2.rectangle(frame,
                          (graph_x, graph_y),
                          (graph_x + self.graph_width, graph_y + self.graph_height),
                          (255, 255, 255), 1)

            points = []
            for i, val in enumerate(normalized):
                x = graph_x + int(i * self.graph_width / len(normalized))
                y = graph_y + self.graph_height - int(val * (self.graph_height - 20)) - 10
                points.append((x, y))

            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points_array], False, (0, 255, 0), 2)

            cv2.putText(frame, "BVP Signal", (graph_x, graph_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_time
        self.last_time = current_time

        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_buffer.append(fps)
            return np.mean(self.fps_buffer)
        return 0

    def run(self):
        print("=" * 50)
        print("Multimodal Stress Detection System")
        print("=" * 50)
        print(f"rPPG Model: {self.rppg.model_name}")
        print(f"FER Model: HSEmotion EfficientNet-B2")
        print("Controls: 'q' quit | 'r' reset | 'l' toggle logging")
        print("=" * 50)

        self.rppg.start_capture(camera_index=0)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                frame, face_box = self.rppg.get_next_frame()

                if frame is None:
                    print("Cannot read frame")
                    break

                h, w = frame.shape[:2]
                frame = cv2.flip(frame, 1)
                flipped_box = self.flip_face_box(face_box, w)

                bpm = self.rppg.update_metrics()
                hrv_metrics = self.rppg.get_hrv_metrics()
                sqi = self.rppg.get_sqi()

                emotion = self.emotion.process_frame(frame, face_box=flipped_box)
                emotion_confidence = self.emotion.get_confidence()

                state, state_color = self.fusion.classify_state(
                    bpm=bpm,
                    emotion=emotion,
                    hrv_rmssd=hrv_metrics.get('rmssd'),
                    hrv_pnn50=hrv_metrics.get('pnn50'),
                    hrv_lf_hf=hrv_metrics.get('lf_hf'),
                    emotion_confidence=emotion_confidence
                )

                hrv_stress_score = None
                if self.fusion._has_hrv_data(
                    hrv_metrics.get('rmssd'),
                    hrv_metrics.get('pnn50'),
                    hrv_metrics.get('lf_hf')
                ):
                    hrv_stress_score = self.fusion._compute_hrv_stress_score(
                        hrv_metrics.get('rmssd'),
                        hrv_metrics.get('pnn50'),
                        hrv_metrics.get('lf_hf')
                    )

                if self.logging_active and self.logger:
                    self.logger.log(
                        bpm=bpm,
                        sqi=sqi,
                        hrv_metrics=hrv_metrics,
                        emotion=emotion,
                        emotion_confidence=emotion_confidence,
                        hrv_stress_score=hrv_stress_score,
                        state=state
                    )

                fps = self.calculate_fps()

                self.draw_face_box(frame, flipped_box)
                self.draw_signal_graph(frame)
                self.draw_ui_overlay(frame, bpm, emotion, emotion_confidence,
                                     state, state_color, fps, sqi, hrv_metrics)

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.rppg.reset()
                    self.emotion.reset()
                elif key == ord('l'):
                    if self.logger:
                        if self.logging_active:
                            self.logger.stop()
                            self.logging_active = False
                        else:
                            self.logger.start()
                            self.logging_active = True

        finally:
            if self.logging_active and self.logger:
                self.logger.stop()
            self.rppg.stop_capture()
            cv2.destroyAllWindows()


def main():
    app = StressDetectionApp(rppg_model='EfficientPhys.rlap')
    app.run()


if __name__ == "__main__":
    main()
