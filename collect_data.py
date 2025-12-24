import pickle
import cv2
import mediapipe as mp
import string

size = 300

class_labels = list(string.ascii_uppercase) + ['space']
number_of_classes = len(class_labels)

labels_dict = {i: class_labels[i] for i in range(number_of_classes)}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)


cap = cv2.VideoCapture(0)
data = []
labels = []

for class_id in range(number_of_classes):
    current_label = labels_dict[class_id]
    print(f'Collecting data for class:{current_label}')
    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if not ret:
            continue

        cv2.putText(frame,f'Class:{current_label}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),2)
        cv2.putText(frame, 'Press "S" to START | "Q" to QUIT',(30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow('frame',frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < size:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            hand_landmarks = results.multi_hand_landmarks[0] 
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(class_id)
                counter += 1

        cv2.putText(frame, f'Class:{current_label}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0, 255, 0), 2)
        cv2.putText(frame, f'Captured:{counter}/{size}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f'Completed class:{current_label}')
cap.release()
cv2.destroyAllWindows()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
