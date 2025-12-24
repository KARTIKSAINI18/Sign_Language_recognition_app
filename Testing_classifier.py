import pickle
import cv2
import mediapipe as mp
import numpy as np
import string

model_dict =pickle.load(open('./model.p','rb'))
model =model_dict['model']

class_labels =list(string.ascii_uppercase) +['space']
labels_dict = {i:class_labels[i] for i in range(len(class_labels))}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands =1,
    min_detection_confidence=0.3
)
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ =[]
    ret, frame = cap.read()
    if not ret:
        continue

    frame= cv2.flip(frame, 1)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks=results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_class = int(prediction[0])
            predicted_label = labels_dict[predicted_class]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            cv2.rectangle(frame,(x1, y1),(x2, y2), (0, 0, 0), 3)
            cv2.putText(
                frame,
                predicted_label,
                (x1, y1 -10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
