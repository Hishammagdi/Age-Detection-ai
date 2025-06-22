import cv2
import numpy as np
import argparse
import os

# Define age buckets
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load age detection model
age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load models
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def detect_and_predict_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        age = AGE_BUCKETS[preds[0].argmax()]
        confidence = preds[0].max() * 100

        label = f"{age} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="Path to image file (optional)")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print("Image file not found.")
            exit(1)
        image = cv2.imread(args.image)
        output = detect_and_predict_age(image)
        cv2.imshow("Age Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access camera.")
            exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = detect_and_predict_age(frame)
            cv2.imshow("Live Age Detection", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
