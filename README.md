# Age-Detection-ai
This project demonstrates **age detection** using deep learning models with **OpenCV and a pre-trained Caffe model**. It uses a deep neural network (DNN) to estimate the age group of people in images or from a webcam.

## 🧠 Description

The application uses OpenCV's DNN module to load and run a pre-trained model for age classification. The model predicts age categories like:


### 🔍 Key Features

- Detect faces in images or webcam feed
- Predict age range using a deep learning model
- Fast and lightweight implementation using OpenCV DNN
- Simple Python script for testing

---

## 📁 Files and Structure



---

## 🧰 Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

### 📦 Install Dependencies

```bash
pip install opencv-python numpy

python age_detector.py --image test.jpg


python age_detector.py
Model Details
Model: age_net.caffemodel

Architecture: age_deploy.prototxt

Input: 227x227 BGR image

Output: Age category with the highest confidence

['(0-2)', '(4-6)', '(8-12)', '(15-20)',
 '(25-32)', '(38-43)', '(48-53)', '(60-100)']

Detected Age: (25-32) with confidence: 95.21%



