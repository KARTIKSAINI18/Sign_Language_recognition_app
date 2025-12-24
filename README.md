# Sign Language Recognition App

This project is a real-time sign language to text recognition system built using computer vision and deep learning.

The main focus of this project is the **process**:
how data is created, how it is trained, and how it is used for real-time prediction.

---

## Data Creation (How Dataset is Made)

- Hand gestures are captured using a webcam.
- MediaPipe is used to detect hand landmarks.
- Landmark coordinates (x, y, z) are extracted for each gesture.
- These landmarks are saved and labeled to form the dataset.
- The dataset contains structured numerical data instead of raw images.

---

## What is Used

- **Python** – core programming language  
- **OpenCV** – webcam access and real-time video processing  
- **MediaPipe** – hand detection and landmark extraction  
- **TensorFlow / Keras** – training the deep learning model  
- **NumPy** – data handling and preprocessing  

---

## What is Done (Working Flow)

1. Webcam captures real-time video.
2. MediaPipe detects the hand and extracts landmarks.
3. The landmarks are passed to the trained model.
4. The model predicts the corresponding sign.
5. The predicted sign is displayed as text on the screen.

---

## How to Run

1. Clone the repository  

2. Install dependencies  

3. Run the application  

---

## Project Scope

- Converts sign language gestures into text
- Works in real-time using a webcam
- Focused on learning and implementation of computer vision + deep learning
