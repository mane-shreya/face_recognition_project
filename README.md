📌 Implementation of PCA with ANN Algorithm for Face Recognition
📖 Project Overview

This project implements a Face Recognition System using Principal Component Analysis (PCA) for feature extraction and an Artificial Neural Network (ANN) for classification.

The system follows the classical Eigenfaces approach proposed by Turk and Pentland (1991) and evaluates performance by varying the number of eigenfaces (k). It also supports unknown (imposter) face detection.

🎯 Objectives

Design a face recognition system using Python

Reduce high-dimensional image data using PCA

Train an ANN (Backpropagation Neural Network) for classification

Analyze how recognition accuracy changes with different values of k

Detect non-enrolled (imposter) faces

📂 Dataset

Source:
https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip

Structure:

dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg


Each subfolder represents one subject, and images inside it are used for training and testing.

🛠 Libraries Used

Only the following libraries are used as per constraints:

NumPy – matrix operations, PCA, eigen decomposition

SciPy – numerical computations

OpenCV (cv2) – image loading, resizing, preprocessing

Matplotlib – plotting accuracy vs k

Scikit-learn – ANN (MLPClassifier), train-test split, accuracy

⚙️ System Architecture
Input Face Image
        ↓
Image Preprocessing (Grayscale + Resize)
        ↓
Mean Face Calculation
        ↓
Mean Zero Normalization
        ↓
PCA (Eigenfaces Generation)
        ↓
Feature Projection
        ↓
ANN Training / Prediction
        ↓
Recognized Face / Unknown

 ANN Training

ANN Type: Backpropagation Neural Network

Hidden Layer: 100 neurons

Dataset split:

60% Training

40% Testing

Input: PCA features

Output: Face class labels

🧪 Testing Phase
1. Test Image Vectorization

Convert test image into a column vector.

2. Mean Zero Alignment
𝐼
2
=
𝐼
1
−
𝑀
I
2
	​

=I
1
	​

−M
3. Projection onto Eigenfaces
Ω
=
Φ
𝐼
2
Ω=ΦI
2
	​

4. Classification

ANN predicts the label

Distance threshold is used to detect unknown (imposter) faces

If distance > threshold → Unknown Person

📊 Performance Evaluation
🔹 Accuracy vs Number of Eigenfaces (k)

The system is evaluated by varying k = {10, 20, 30, 40, 50, 60, 70}

Observation:

Accuracy improves as k increases

Best performance observed around k = 50–70

Too small k → loss of discriminative information

📈 Accuracy vs k Plot:

🔹 Sample Prediction Output

🚫 Imposter Detection

Imposters (not present in training set) are added to test data

If projected face distance exceeds a threshold → classified as “Unknown”

▶️ How to Run the Project
# 1. Clone repository
git clone <repo-link>

# 2. Place dataset in project directory
dataset/

# 3. Run the main script
python main.py


Outputs generated:

accuracy_vs_k.png

predicted_output.jpg

✅ Results Summary

PCA effectively reduces dimensionality

ANN improves classification accuracy

Optimal k value significantly impacts performance

System successfully detects unknown faces

📚 Reference

Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience.

👩‍💻 Author

Shreya Mane
Final Year Student – Computer Science
Project: PCA with ANN for Face Recognition
