import sys
import cv2
import pickle
import time
import string
import mediapipe as mp
import numpy as np
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, Qt


def apply_shadow(widget, blur=35, x=0, y=10, alpha=120):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)

    shadow.setXOffset(x)
    shadow.setYOffset(y)

    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language to Text")


        self.setGeometry(100, 100, 1250, 720)


        model_path = resource_path("model.p")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)["model"]


        self.class_labels = list(string.ascii_uppercase) + ["space"]

        self.labels_dict = {i: self.class_labels[i] for i in range(len(self.class_labels))}

        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3
        )

        
        self.final_text = ""
        self.current_letter = None

        self.start_time = None
        self.HOLD_TIME = 3
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)

        main_layout.setSpacing(25)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(18)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)

        self.camera_label.setAlignment(Qt.AlignCenter)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Predicted text will appear here…")
        self.output_text.setFixedHeight(170)

        self.status_label = QLabel("✋ Waiting for gesture")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.status_label.setObjectName("statusLabel")

        self.clear_btn = QPushButton("Clear")
        self.delete_btn = QPushButton("Delete")

        self.delete_btn.setObjectName("deleteBtn")

        self.clear_btn.clicked.connect(self.clear_text)

        self.delete_btn.clicked.connect(self.delete_char)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.delete_btn)

        left_layout.addWidget(self.camera_label)
        right_layout.addWidget(self.output_text)

        right_layout.addWidget(self.status_label)

        right_layout.addLayout(btn_layout)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        central.setLayout(main_layout)

        apply_shadow(self.camera_label)
        apply_shadow(self.output_text)

        apply_shadow(self.status_label)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def clear_text(self):
        self.final_text = ""
        self.output_text.setText("")

    def delete_char(self):
        self.final_text = self.final_text[:-1]
        self.output_text.setText(self.final_text)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        predicted_label = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            x_, y_, data_aux = [], [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            if len(data_aux) == 42:
                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_label = self.labels_dict[int(prediction[0])]

        if predicted_label:
            if predicted_label == self.current_letter:
                elapsed = time.time() - self.start_time

                self.status_label.setText(
                    f"Holding {predicted_label}  ({elapsed:.1f}/{self.HOLD_TIME}s)"
                )
                self.status_label.setStyleSheet("color:#38bdf8;")
                if elapsed >= self.HOLD_TIME:
                    self.final_text += " " if predicted_label == "space" else predicted_label
                    self.output_text.setText(self.final_text)

                    self.status_label.setStyleSheet("color:#22c55e;")
                    self.current_letter = None
                    self.start_time = None
            else:
                self.current_letter = predicted_label
                self.start_time = time.time()

        else:
            self.status_label.setText("✋ Waiting for gesture")
            self.status_label.setStyleSheet("color:#94a3b8;")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)

        self.camera_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.cap.release()
        event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    qss_path = resource_path(os.path.join("ui", "styles.qss"))

    with open(qss_path, "r") as f:
        app.setStyleSheet(f.read())

    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())
