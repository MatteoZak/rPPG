
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from fusion_engine import FusionEngine

def _fill_buffer(fusion, bpm, emotion, n=15, **kwargs):
    state, color = None, None
    for _ in range(n):
        state, color = fusion.classify_state(bpm, emotion, **kwargs)
    return state, color

class TestClassifyStateIntegration:

    def setup_method(self):
        self.fusion = FusionEngine()

    def test_zero_bpm_returns_unknown(self):
        state, _ = _fill_buffer(self.fusion, 0, 'Neutral', n=15)
        assert state == "UNKNOWN"

    def test_negative_bpm_returns_unknown(self):
        state, _ = _fill_buffer(self.fusion, -1, 'Neutral', n=15)
        assert state == "UNKNOWN"

    def test_emotion_confidence_below_threshold_treated_as_neutral(self):
        state, _ = _fill_buffer(
            self.fusion, 100, 'Happy', n=15,
            emotion_confidence=20,
            hrv_rmssd=15, hrv_pnn50=3, hrv_lf_hf=3.5
        )
        assert state == "HIGH STRESS"

    def test_emotion_confidence_above_threshold_respected(self):
        state, _ = _fill_buffer(
            self.fusion, 100, 'Happy', n=15,
            emotion_confidence=50
        )
        assert state == "EXCITEMENT"

    def test_emotion_confidence_none_does_not_override(self):
        state, _ = _fill_buffer(
            self.fusion, 100, 'Happy', n=15,
            emotion_confidence=None
        )
        assert state == "EXCITEMENT"

    # --- Hysteresis ---

    def test_hysteresis_lowers_threshold_when_in_high_state(self):
        self.fusion.last_stable_state = "HIGH STRESS"
        state, _ = _fill_buffer(self.fusion, 87, 'Neutral', n=15)
        assert state == "ALERT"

    def test_hysteresis_raises_threshold_when_in_relaxed_state(self):
        self.fusion.last_stable_state = "RELAXED"
        state, _ = _fill_buffer(self.fusion, 92, 'Neutral', n=15)
        assert state == "RELAXED"

    def test_hysteresis_for_excitement_state(self):
        self.fusion.last_stable_state = "EXCITEMENT"
        state, _ = _fill_buffer(self.fusion, 87, 'Happy', n=15)
        assert state == "EXCITEMENT"

    def test_hysteresis_for_alert_state(self):
        self.fusion.last_stable_state = "ALERT"
        state, _ = _fill_buffer(self.fusion, 87, 'Fear', n=15)
        assert state == "ALERT"

    def test_hysteresis_for_fatigue_state(self):
        self.fusion.last_stable_state = "FATIGUE"
        state, _ = _fill_buffer(self.fusion, 92, 'Neutral', n=15)
        assert state == "RELAXED"

    def test_state_buffer_requires_minimum_10_frames(self):
        for _ in range(9):
            state, _ = self.fusion.classify_state(70, 'Neutral')
        assert state == "UNKNOWN"

    def test_state_buffer_changes_at_10_frames_with_majority(self):
        state, _ = _fill_buffer(self.fusion, 70, 'Neutral', n=10)
        assert state == "RELAXED"

    def test_state_buffer_70_percent_majority_needed(self):
        fusion = FusionEngine()
        for _ in range(7):
            fusion.classify_state(70, 'Neutral')
        for _ in range(3):
            state, _ = fusion.classify_state(70, 'Fear')
        assert state == "RELAXED"

    def test_state_buffer_insufficient_majority(self):
        fusion = FusionEngine()
        for _ in range(6):
            fusion.classify_state(70, 'Neutral')
        for _ in range(4):
            state, _ = fusion.classify_state(70, 'Fear')
        assert state == "UNKNOWN"

    def test_state_color_high_stress(self):
        _, color = _fill_buffer(
            self.fusion, 100, 'Neutral', n=15,
            hrv_rmssd=15, hrv_pnn50=3, hrv_lf_hf=3.5
        )
        assert color == (0, 0, 255)

    def test_state_color_relaxed(self):
        _, color = _fill_buffer(self.fusion, 70, 'Neutral', n=15)
        assert color == (0, 255, 0)

    def test_end_to_end_stressed_person(self):
        state, color = _fill_buffer(
            self.fusion, 105, 'Neutral', n=15,
            emotion_confidence=60,
            hrv_rmssd=18, hrv_pnn50=4, hrv_lf_hf=3.2
        )
        assert state == "HIGH STRESS"
        assert color == (0, 0, 255)

    def test_end_to_end_relaxed_person(self):
        state, color = _fill_buffer(
            self.fusion, 75, 'Happy', n=15,
            emotion_confidence=85,
            hrv_rmssd=52, hrv_pnn50=22, hrv_lf_hf=0.9
        )
        assert state == "RELAXED"
        assert color == (0, 255, 0)

    def test_end_to_end_fatigued_person(self):
        state, color = _fill_buffer(
            self.fusion, 62, 'Neutral', n=15,
            emotion_confidence=55,
            hrv_rmssd=38, hrv_pnn50=14, hrv_lf_hf=2.8
        )
        assert state == "FATIGUE"
        assert color == (255, 100, 0)


class TestFusionEngineAccessors:

    def test_initial_state_is_unknown(self):
        assert FusionEngine().get_state() == "UNKNOWN"

    def test_initial_color_is_gray(self):
        assert FusionEngine().get_color() == (128, 128, 128)

    def test_set_bpm_threshold(self):
        f = FusionEngine()
        f.set_bpm_threshold(100)
        assert f.bpm_threshold == 100

    def test_default_bpm_threshold(self):
        assert FusionEngine().bpm_threshold == 90

    def test_default_hysteresis(self):
        assert FusionEngine().hysteresis == 5

    def test_default_min_emotion_confidence(self):
        assert FusionEngine().min_emotion_confidence == 30.0

    def test_custom_initialization(self):
        f = FusionEngine(bpm_threshold=80, hysteresis=10, min_emotion_confidence=50)
        assert f.bpm_threshold == 80
        assert f.hysteresis == 10
        assert f.min_emotion_confidence == 50


def _create_mock_emotion_engine(**kwargs):
    with patch('emotion_engine.HSEmotionRecognizer') as mock_fer, \
         patch('emotion_engine.cv2') as mock_cv2, \
         patch('emotion_engine.torch') as mock_torch:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.load = MagicMock(return_value=MagicMock())
        mock_torch.device.return_value = 'cpu'
        mock_cv2.CascadeClassifier.return_value = MagicMock()
        mock_cv2.data.haarcascades = ''
        from emotion_engine import EmotionEngine
        engine = EmotionEngine(**kwargs)
    return engine


class TestEmotionEngineMapping:

    def setup_method(self):
        self.engine = _create_mock_emotion_engine()

    def test_anger_maps_to_angry(self):
        assert self.engine.emotion_map['Anger'] == 'Angry'

    def test_contempt_maps_to_angry(self):
        assert self.engine.emotion_map['Contempt'] == 'Angry'

    def test_disgust_maps_to_angry(self):
        assert self.engine.emotion_map['Disgust'] == 'Angry'

    def test_fear_maps_to_fear(self):
        assert self.engine.emotion_map['Fear'] == 'Fear'

    def test_happiness_maps_to_happy(self):
        assert self.engine.emotion_map['Happiness'] == 'Happy'

    def test_neutral_maps_to_neutral(self):
        assert self.engine.emotion_map['Neutral'] == 'Neutral'

    def test_sadness_maps_to_sad(self):
        assert self.engine.emotion_map['Sadness'] == 'Sad'

    def test_surprise_maps_to_surprise(self):
        assert self.engine.emotion_map['Surprise'] == 'Surprise'

    def test_all_eight_classes_covered(self):
        expected_keys = {'Anger', 'Contempt', 'Disgust', 'Fear',
                         'Happiness', 'Neutral', 'Sadness', 'Surprise'}
        assert set(self.engine.emotion_map.keys()) == expected_keys


class TestEmotionEngineReset:

    def setup_method(self):
        self.engine = _create_mock_emotion_engine()

    def test_reset_clears_emotion(self):
        self.engine.current_emotion = 'Angry'
        self.engine.reset()
        assert self.engine.current_emotion == 'Neutral'

    def test_reset_clears_confidence(self):
        self.engine.emotion_confidence = 85.0
        self.engine.reset()
        assert self.engine.emotion_confidence == 0.0

    def test_reset_clears_scores(self):
        self.engine.emotion_scores = {'Anger': 0.8, 'Neutral': 0.2}
        self.engine.reset()
        assert self.engine.emotion_scores == {}

    def test_reset_clears_frame_counter(self):
        self.engine.frame_counter = 10
        self.engine.reset()
        assert self.engine.frame_counter == 0


def _create_mock_rppg_processor(**kwargs):
    with patch('rppg_processor.rppg.Model') as mock_model_class:
        mock_model_class.return_value = MagicMock()
        from rppg_processor import RPPGProcessor
        processor = RPPGProcessor(**kwargs)
    return processor


class TestRPPGProcessorMetricStructure:

    def setup_method(self):
        self.processor = _create_mock_rppg_processor()

    def test_hrv_metrics_has_all_required_keys(self):
        metrics = self.processor.get_hrv_metrics()
        required_keys = {'sdnn', 'rmssd', 'pnn50', 'lf_hf', 'breathing_rate'}
        assert set(metrics.keys()) == required_keys

    def test_initial_hrv_metrics_are_zero(self):
        metrics = self.processor.get_hrv_metrics()
        for key, value in metrics.items():
            assert value == 0.0, f"{key} should be 0.0 initially, got {value}"

    def test_initial_bpm_is_zero(self):
        assert self.processor.current_bpm == 0

    def test_initial_sqi_is_zero(self):
        assert self.processor.get_sqi() == 0.0


class TestRPPGProcessorReset:

    def setup_method(self):
        self.processor = _create_mock_rppg_processor()

    def test_reset_clears_bpm(self):
        self.processor.current_bpm = 72
        self.processor.reset()
        assert self.processor.current_bpm == 0

    def test_reset_clears_sqi(self):
        self.processor.current_sqi = 0.8
        self.processor.reset()
        assert self.processor.current_sqi == 0.0

    def test_reset_clears_all_hrv(self):
        self.processor.current_rmssd = 45.0
        self.processor.current_sdnn = 50.0
        self.processor.current_pnn50 = 20.0
        self.processor.current_lf_hf = 1.5
        self.processor.current_breathing_rate = 15.0
        self.processor.reset()
        metrics = self.processor.get_hrv_metrics()
        for key, value in metrics.items():
            assert value == 0.0, f"{key} should be 0.0 after reset"

    def test_reset_clears_hr_timer(self):
        self.processor._last_hr_time = 999999
        self.processor.reset()
        assert self.processor._last_hr_time == 0

    def test_reset_preserves_capture_state(self):
        self.processor._is_capturing = True
        self.processor.reset()
        assert self.processor._is_capturing is True


class TestParametrizedStressScenarios:

    @pytest.mark.parametrize(
        "scenario, bpm, emotion, confidence, rmssd, pnn50, lf_hf, expected_state",
        [
            pytest.param(
                "Exam anxiety (suppressed stress)",
                105, 'Neutral', 60, 18, 4, 3.2, "HIGH STRESS",
                id="exam_anxiety"
            ),
            pytest.param(
                "Watching a comedy",
                75, 'Happy', 85, 52, 22, 0.9, "RELAXED",
                id="watching_comedy"
            ),
            pytest.param(
                "Jump scare in a video game",
                110, 'Fear', 70, 25, 8, 2.8, "ALERT",
                id="jump_scare"
            ),
            pytest.param(
                "Winning a prize (positive excitement)",
                102, 'Surprise', 75, 40, 18, 1.5, "EXCITEMENT",
                id="winning_prize"
            ),
            pytest.param(
                "Late night studying (fatigue)",
                62, 'Neutral', 55, 38, 14, 2.8, "FATIGUE",
                id="late_night_studying"
            ),
            pytest.param(
                "Post-exercise cool-down (positive excitement)",
                98, 'Happy', 90, 45, 20, 1.2, "EXCITEMENT",
                id="post_exercise"
            ),
            pytest.param(
                "Argument with someone",
                108, 'Angry', 65, 22, 6, 3.0, "ALERT",
                id="argument"
            ),
            pytest.param(
                "Crying after bad news",
                68, 'Sad', 80, 24, 7, 2.5, "FATIGUE",
                id="crying_bad_news"
            ),
            pytest.param(
                "Meditation session",
                58, 'Neutral', 45, 60, 30, 0.7, "RELAXED",
                id="meditation"
            ),
            pytest.param(
                "System just started (no data yet)",
                0, 'Neutral', 0, 0, 0, 0, "UNKNOWN",
                id="no_data"
            ),
        ],
    )
    def test_scenario(self, scenario, bpm, emotion, confidence, rmssd,
                      pnn50, lf_hf, expected_state):
        fusion = FusionEngine()
        state, _ = _fill_buffer(
            fusion, bpm, emotion, n=15,
            emotion_confidence=confidence,
            hrv_rmssd=rmssd, hrv_pnn50=pnn50, hrv_lf_hf=lf_hf
        )
        assert state == expected_state, (
            f"Scenario '{scenario}': expected {expected_state}, got {state}"
        )
