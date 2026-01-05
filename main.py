import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ========== STEP 1: Load Images and Labels ==========
def load_images_and_labels(folder, size=(100, 100)):
    images, labels = [], []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, size)
                images.append(img.flatten())
                labels.append(os.path.basename(root))
    return np.array(images), np.array(labels)

# ========== STEP 2: PCA ==========
def compute_pca(X, k):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    cov = np.dot(X_centered, X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1][:k]]
    eigenfaces = np.dot(X_centered.T, eigvecs).T
    X_pca = np.dot(X_centered, eigenfaces.T)
    return X_pca, eigenfaces, mean_face

# ========== STEP 3: Predict New Image (with Unknown Detection) ==========
def predict_new_image(image_path, eigenfaces, mean_face, model, X_train_pca, y_train, size=(100, 100), threshold=1000):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_gray is None or img_color is None:
        print("❌ Image read error.")
        return

    img_gray = cv2.resize(img_gray, size)
    img_color = cv2.resize(img_color, size)
    img_flat = img_gray.flatten().astype(np.float32)
    img_centered = img_flat - mean_face
    projected = np.dot(eigenfaces, img_centered)

    pred_label = model.predict([projected])[0]
    pred_proba = model.predict_proba([projected])
    max_conf = np.max(pred_proba)

    # Unknown Detection via distance to nearest known vector
    min_dist = np.min(np.linalg.norm(X_train_pca - projected, axis=1))

    if min_dist > threshold:
        pred_label = "Unknown"

    print(f"✅ Predicted: {pred_label} (confidence: {max_conf:.2f}, distance: {min_dist:.2f})")

    out_img = cv2.putText(img_color, f"Predicted: {pred_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("predicted_output.jpg", out_img)
    print("📷 Saved: predicted_output.jpg")

# ========== STEP 4: Accuracy vs k ==========
def evaluate_k(X, y, k_list):
    accuracies = []
    for k in k_list:
        print(f"\n🔁 Running PCA + ANN for k = {k}")
        X_pca, _, _ = compute_pca(X, k)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        accuracies.append(acc)

    plt.plot(k_list, accuracies, marker='o')
    plt.xlabel("Number of Eigenfaces (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Eigenfaces (k)")
    plt.grid(True)
    plt.savefig("accuracy_vs_k.png")
    print("📊 Saved: accuracy_vs_k.png")
    plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    dataset_folder = "dataset"
    test_image_path = "test_face.jpg"
    k_final = 50

    X, y = load_images_and_labels(dataset_folder)
    if len(X) == 0:
        print("❌ No images loaded.")
        exit()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_pca, eigenfaces, mean_face = compute_pca(X, k=k_final)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict with unknown detection
    predict_new_image(test_image_path, eigenfaces, mean_face, clf, X_train, y_train, threshold=1000)

    # Accuracy vs. k
    evaluate_k(X, y_encoded, k_list=[10, 20, 30, 40, 50, 60, 70])
