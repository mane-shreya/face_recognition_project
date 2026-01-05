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

🧠 Training Phase (Step-by-Step)
1. Face Database Generation

Each image is converted into a column vector.

Image size: m × n

Total images: p

Face database size:

𝐹
𝑎
𝑐
𝑒
_
𝐷
𝑏
∈
𝑅
(
𝑚
𝑛
)
×
𝑝
Face_Db∈R
(mn)×p
2. Mean Face Calculation

The mean face is computed as:

𝑀
=
1
𝑝
∑
𝑖
=
1
𝑝
𝐹
𝑎
𝑐
𝑒
_
𝐷
𝑏
(
:
,
𝑖
)
M=
p
1
	​

i=1
∑
p
	​

Face_Db(:,i)

Dimension:

𝑀
∈
𝑅
(
𝑚
𝑛
)
×
1
M∈R
(mn)×1
3. Mean Zero Normalization

Subtract the mean face from each image:

Δ
𝑖
=
𝐹
𝑎
𝑐
𝑒
_
𝐷
𝑏
(
:
,
𝑖
)
−
𝑀
Δ
i
	​

=Face_Db(:,i)−M
4. Surrogate Covariance Matrix

Instead of computing a large 
(
𝑚
𝑛
×
𝑚
𝑛
)
(mn×mn) covariance matrix, Turk & Pentland’s surrogate covariance is used:

𝐶
=
Δ
𝑇
Δ
C=Δ
T
Δ

Dimension:

𝐶
∈
𝑅
𝑝
×
𝑝
C∈R
p×p

This significantly reduces computational complexity.

5. Eigen Decomposition

Compute eigenvalues and eigenvectors of the covariance matrix:

𝐶
𝑉
=
𝜆
𝑉
CV=λV

Eigenvectors sorted in descending order of eigenvalues

6. Selection of k Best Directions

Select top k eigenvectors corresponding to the largest eigenvalues:

Ψ
∈
𝑅
𝑝
×
𝑘
Ψ∈R
p×k
7. Eigenfaces Generation

Project mean-aligned faces onto feature vectors:

Φ
=
Ψ
𝑇
Δ
𝑇
Φ=Ψ
T
Δ
T

Eigenfaces dimension:

Φ
∈
𝑅
𝑘
×
𝑚
𝑛
Φ∈R
k×mn
8. Face Signature Generation

Each face is represented as a signature vector:

𝜔
𝑖
=
Φ
Δ
𝑖
ω
i
	​

=ΦΔ
i
	​


Signature matrix size:

𝜔
∈
𝑅
𝑘
×
𝑝
ω∈R
k×p
9. ANN Training

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
