from collections import deque


class FusionEngine:

    def __init__(self, bpm_threshold=90, hysteresis=5, min_emotion_confidence=30.0):
        self.bpm_threshold = bpm_threshold
        self.hysteresis = hysteresis
        self.min_emotion_confidence = min_emotion_confidence
        self.current_state = "UNKNOWN"
        self.state_color = (128, 128, 128)

        self.state_buffer = deque(maxlen=15)
        self.last_stable_state = "UNKNOWN"

        self.state_colors = {
            "HIGH STRESS": (0, 0, 255),
            "EXCITEMENT": (0, 165, 255),
            "RELAXED": (0, 255, 0),
            "ALERT": (0, 140, 255),
            "FATIGUE": (255, 100, 0),
            "UNKNOWN": (128, 128, 128)
        }

    def classify_state(self, bpm, emotion, hrv_rmssd=None, hrv_pnn50=None,
                       hrv_lf_hf=None, emotion_confidence=None):
        if emotion_confidence is not None and emotion_confidence < self.min_emotion_confidence:
            emotion = 'Neutral'

        if bpm <= 0:
            tentative_state = "UNKNOWN"
        else:
            if self.last_stable_state in ["HIGH STRESS", "EXCITEMENT", "ALERT"]:
                effective_threshold = self.bpm_threshold - self.hysteresis
            else:
                effective_threshold = self.bpm_threshold + self.hysteresis

            hrv_available = self._has_hrv_data(hrv_rmssd, hrv_pnn50, hrv_lf_hf)

            if hrv_available:
                hrv_stress_score = self._compute_hrv_stress_score(hrv_rmssd, hrv_pnn50, hrv_lf_hf)
                tentative_state = self._classify_with_hrv(
                    bpm, emotion, effective_threshold, hrv_stress_score, hrv_lf_hf)
            else:
                tentative_state = self._classify_without_hrv(
                    bpm, emotion, effective_threshold)

        self.state_buffer.append(tentative_state)

        if len(self.state_buffer) >= 10:
            state_counts = {}
            for state in self.state_buffer:
                state_counts[state] = state_counts.get(state, 0) + 1

            most_common_state = max(state_counts, key=state_counts.get)
            most_common_count = state_counts[most_common_state]

            if most_common_count >= len(self.state_buffer) * 0.7:
                self.last_stable_state = most_common_state
                self.current_state = most_common_state
                self.state_color = self.state_colors[most_common_state]
        else:
            self.current_state = self.last_stable_state
            self.state_color = self.state_colors[self.last_stable_state]

        return self.current_state, self.state_color

    def _has_hrv_data(self, rmssd, pnn50, lf_hf):
        if rmssd is not None and rmssd > 0:
            return True
        if pnn50 is not None and pnn50 > 0:
            return True
        if lf_hf is not None and lf_hf > 0:
            return True
        return False

    def _classify_with_hrv(self, bpm, emotion, threshold, hrv_stress_score, hrv_lf_hf):
        if bpm > threshold:
            if emotion == 'Neutral':
                if hrv_stress_score > 0.6:
                    return "HIGH STRESS"
                elif hrv_stress_score < 0.3:
                    return "EXCITEMENT"
                else:
                    return "ALERT"

            elif emotion in ['Happy', 'Surprise']:
                return "EXCITEMENT"

            elif emotion in ['Fear', 'Angry']:
                return "ALERT"

            else:
                if hrv_stress_score > 0.5:
                    return "HIGH STRESS"
                else:
                    return "ALERT"
        else:
            if emotion in ['Neutral', 'Happy']:
                if hrv_stress_score > 0.7:
                    return "HIGH STRESS"
                elif hrv_stress_score > 0.4 and emotion == 'Neutral':
                    if hrv_lf_hf is not None and hrv_lf_hf > 2.5:
                        return "FATIGUE"
                    else:
                        return "RELAXED"
                else:
                    return "RELAXED"

            elif emotion == 'Sad':
                if hrv_stress_score > 0.5:
                    return "FATIGUE"
                else:
                    return "RELAXED"

            elif emotion in ['Fear', 'Angry']:
                return "ALERT"

            else:
                return "RELAXED"

    def _classify_without_hrv(self, bpm, emotion, threshold):
        if bpm > threshold:
            if emotion in ['Happy', 'Surprise']:
                return "EXCITEMENT"
            elif emotion in ['Fear', 'Angry']:
                return "ALERT"
            elif emotion == 'Neutral':
                return "ALERT"
            else:
                return "ALERT"
        else:
            if emotion in ['Neutral', 'Happy']:
                return "RELAXED"
            elif emotion == 'Sad':
                return "FATIGUE"
            elif emotion in ['Fear', 'Angry']:
                return "ALERT"
            else:
                return "RELAXED"

    def _compute_hrv_stress_score(self, rmssd, pnn50, lf_hf):
        scores = []
        weights = []

        if rmssd is not None and rmssd > 0:
            rmssd_score = max(0.0, min(1.0, 1.0 - (rmssd - 20) / 30.0))
            scores.append(rmssd_score)
            weights.append(0.4)

        if pnn50 is not None and pnn50 >= 0:
            pnn50_score = max(0.0, min(1.0, 1.0 - (pnn50 - 5) / 20.0))
            scores.append(pnn50_score)
            weights.append(0.25)

        if lf_hf is not None and lf_hf > 0:
            lf_hf_score = max(0.0, min(1.0, (lf_hf - 1.0) / 2.0))
            scores.append(lf_hf_score)
            weights.append(0.35)

        if not scores:
            return 0.5

        total_weight = sum(weights)
        stress_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return stress_score

    def get_state(self):
        return self.current_state

    def get_color(self):
        return self.state_color

    def set_bpm_threshold(self, threshold):
        self.bpm_threshold = threshold
