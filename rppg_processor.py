import rppg
import numpy as np
import time
import cv2


class RPPGProcessor:

    def __init__(self, model_name='EfficientPhys.pure', fps=30):
        self.model_name = model_name
        self.fps = fps
        self.model = rppg.Model(model_name)

        self.current_bpm = 0
        self.current_sqi = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_pnn50 = 0.0
        self.current_lf_hf = 0.0
        self.current_breathing_rate = 0.0

        self._last_hr_time = 0
        self._hr_update_interval = 1.0

        self._capture_context = None
        self._preview_iter = None
        self._is_capturing = False

    def start_capture(self, camera_index=0):
        self._capture_context = self.model.video_capture(camera_index)
        self._preview_iter = iter(self.model.preview)
        self._is_capturing = True

    def stop_capture(self):
        if self._capture_context is not None:
            try:
                self._capture_context.__exit__(None, None, None)
            except Exception:
                pass
            self._capture_context = None
            self._preview_iter = None
            self._is_capturing = False

    def get_next_frame(self):
        if not self._is_capturing or self._preview_iter is None:
            return None, None

        try:
            frame_rgb, box = next(self._preview_iter)

            if frame_rgb is None:
                return None, None

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr, box

        except StopIteration:
            self._is_capturing = False
            return None, None

    def update_metrics(self):
        now = time.time()
        if now - self._last_hr_time < self._hr_update_interval:
            return self.current_bpm

        self._last_hr_time = now

        if not self.model.has_signal:
            return 0

        try:
            result = self.model.hr(start=-15)

            if result is None:
                return self.current_bpm

            hr = result.get('hr')
            if hr is not None and not np.isnan(hr) and hr > 0:
                self.current_bpm = float(hr)

            sqi = result.get('SQI')
            if sqi is not None and not np.isnan(sqi):
                self.current_sqi = float(sqi)

            hrv = result.get('hrv', {})
            if hrv:
                sdnn = hrv.get('sdnn', 0)
                if sdnn is not None and not np.isnan(sdnn):
                    self.current_sdnn = float(sdnn)

                rmssd = hrv.get('rmssd', 0)
                if rmssd is not None and not np.isnan(rmssd):
                    self.current_rmssd = float(rmssd)

                pnn50 = hrv.get('pnn50', 0)
                if pnn50 is not None and not np.isnan(pnn50):
                    self.current_pnn50 = float(pnn50)

                lf_hf = hrv.get('LF/HF', 0)
                if lf_hf is not None and not np.isnan(lf_hf):
                    self.current_lf_hf = float(lf_hf)

                breathing_rate = hrv.get('breathingrate', 0)
                if breathing_rate is not None and not np.isnan(breathing_rate):
                    self.current_breathing_rate = float(breathing_rate)

        except Exception as e:
            print(f"[rPPG] Error: {e}")

        return self.current_bpm

    def get_filtered_signal(self):
        if not self.model.has_signal:
            return np.array([])

        try:
            bvp_signal, timestamps = self.model.bvp(start=-15)
            if len(bvp_signal) > 0:
                return np.array(bvp_signal)
        except Exception:
            pass

        return np.array([])

    def get_hrv_metrics(self):
        return {
            'sdnn': self.current_sdnn,
            'rmssd': self.current_rmssd,
            'pnn50': self.current_pnn50,
            'lf_hf': self.current_lf_hf,
            'breathing_rate': self.current_breathing_rate
        }

    def get_sqi(self):
        return self.current_sqi

    def reset(self):
        self.current_bpm = 0
        self.current_sqi = 0.0
        self.current_sdnn = 0.0
        self.current_rmssd = 0.0
        self.current_pnn50 = 0.0
        self.current_lf_hf = 0.0
        self.current_breathing_rate = 0.0
        self._last_hr_time = 0
