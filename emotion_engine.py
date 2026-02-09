import cv2
import numpy as np
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionEngine:

    def __init__(self, detection_interval=15, device=None, debug=False):
        self.detection_interval = detection_interval
        self.frame_counter = 0
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.emotion_scores = {}
        self._debug = debug

        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'cpu'
                self._torch_device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'gpu'
                self._torch_device = 'cuda'
            else:
                self.device = 'cpu'
                self._torch_device = 'cpu'
        else:
            self.device = 'gpu' if device in ('cuda', 'mps') else 'cpu'
            self._torch_device = device

        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False, 'map_location': torch.device('cpu')})
        self.model_name = 'enet_b2_8'
        self.fer = HSEmotionRecognizer(model_name=self.model_name, device=self.device)
        torch.load = _original_load

        self.emotion_map = {
            'Anger': 'Angry',
            'Contempt': 'Angry',
            'Disgust': 'Angry',
            'Fear': 'Fear',
            'Happiness': 'Happy',
            'Neutral': 'Neutral',
            'Sadness': 'Sad',
            'Surprise': 'Surprise'
        }

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _crop_face_from_box(self, frame, face_box):
        if face_box is None:
            return None

        h, w = frame.shape[:2]
        y1, y2 = int(face_box[0][0]), int(face_box[0][1])
        x1, x2 = int(face_box[1][0]), int(face_box[1][1])

        face_w = x2 - x1
        face_h = y2 - y1

        if face_w <= 0 or face_h <= 0:
            return None

        pad_w = int(face_w * 0.4)
        pad_h_top = int(face_h * 0.3)
        pad_h_bottom = int(face_h * 0.5)

        y1 = max(0, y1 - pad_h_top)
        y2 = min(h, y2 + pad_h_bottom)
        x1 = max(0, x1 - pad_w)
        x2 = min(w, x2 + pad_w)

        if x2 - x1 < 48 or y2 - y1 < 48:
            return None

        face_crop = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return face_rgb

    def _detect_face_fallback(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        face_crop = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return face_rgb

    def process_frame(self, frame, face_box=None):
        self.frame_counter += 1

        if self.frame_counter >= self.detection_interval:
            self.frame_counter = 0
            self._detect_emotion(frame, face_box)

        return self.current_emotion

    def _detect_emotion(self, frame, face_box=None):
        try:
            face_img = self._crop_face_from_box(frame, face_box)
            if face_img is None:
                face_img = self._detect_face_fallback(frame)

            if face_img is None:
                return

            emotion_label, scores = self.fer.predict_emotions(face_img, logits=False)

            self.current_emotion = self.emotion_map.get(emotion_label, 'Neutral')
            self.emotion_confidence = float(np.max(scores)) * 100.0

            emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear',
                              'Happiness', 'Neutral', 'Sadness', 'Surprise']
            self.emotion_scores = dict(zip(emotion_labels, scores))

        except Exception as e:
            print(f"[EmotionEngine] Error: {e}")

    def get_emotion(self):
        return self.current_emotion

    def get_confidence(self):
        return self.emotion_confidence

    def get_emotion_scores(self):
        return self.emotion_scores

    def reset(self):
        self.frame_counter = 0
        self.current_emotion = "Neutral"
        self.emotion_confidence = 0.0
        self.emotion_scores = {}
