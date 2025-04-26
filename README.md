# Face Recognition Project

## Overview
This project implements a **Face Recognition System** using PyTorch, designed to identify or verify individuals based on facial images. The system addresses challenges like lighting variations, facial expressions, and occlusions, making it suitable for real-world applications such as biometric authentication, surveillance, and personalized user experiences.

---

## Features
- **Custom Dataset Handling**: Automatically processes datasets with subdirectories for each class.
- **Dynamic Class Detection**: Automatically determines the number of classes based on the dataset structure.
- **Model Training**: Includes a training pipeline with loss tracking and accuracy calculation.
- **Inference**: Predicts the class of unseen images or directories of images.
- **Robustness**: Handles variations in lighting, expressions, and occlusions.

---

## Input-Output
- **Input**: A color image (in `.jpg` or `.png` format) of a person’s face.
- **Output**: A class label identifying the individual or a similarity score for verification.

### Example:
- **Input**: A facial image of an individual.
- **Output**: A class label like "Person A" or a similarity score indicating how closely the input matches stored reference images.

---

## Dataset
- **Labeled Faces in the Wild (LFW)**: Over 13,000 labeled face images with varied lighting and expressions. 

[Click here to download the Labeled Faces in the Wild (LFW) dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)


---

## Model Architecture
The model is based on a **Convolutional Neural Network (CNN)**, which is effective for image-related tasks. Key components:
1. **Input Layer**: Normalizes RGB face images.
2. **Feature Extraction**: Extracts facial features using convolution, ReLU activation, and pooling.
3. **Embedding Layer**: Converts features into compact face representations.
4. **Classification & Matching**:
   - **Identification**: Uses Softmax for classification.
   - **Verification**: Compares embeddings using similarity measures.

---

## Training
1. **Run Training**: 
   ```bash
   python train.py

---

## Inference
1. **Run Prediction**:
   ```bash
   python predict.py
   ```
2. **Output**:
   - Predicts the class of images in a directory.
   - Example:
     ```
     Image: ./data/test_data/Tony_Blair_0002.jpg, Prediction: Person A
     Image: ./data/test_data/Colin_Powell_0001.jpg, Prediction: Person B
     ```

---

## Conclusion
This project demonstrates a robust **CNN-based face recognition system** capable of accurate, real-time identification and verification. It is designed to handle real-world challenges such as lighting variations, facial expressions, and occlusions. The system is suitable for applications like:

- **Biometric Authentication**: Secure access control for devices and systems.
- **Surveillance Systems**: Real-time monitoring and identification.
- **Personalized User Experiences**: Tailored interactions in consumer devices.

By leveraging deep learning and publicly available datasets, this project provides a scalable and efficient solution for face recognition tasks in various domains.