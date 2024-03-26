#Hazırlayan: Furkan Yorgun

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data_folder = "C:/Users/FURKAN/Desktop/BitirmeProjem/data/train"

# Sınıf etiketlerini ve görüntüleri depolamak için listeler oluşturalım
images = []
labels = []

# Her sınıf için belirli sayıda görüntü kullanmak için bir sayaç oluşturduk
image_counter = {}

# Klasörleri gezinerek görüntüleri ve sınıfları topluyoruz
for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)
    if os.path.isdir(class_path):
        image_counter[class_folder] = 0  # Her sınıf için başlangıç sayısını sıfıra ayarla
        for image_file in os.listdir(class_path):
            if image_counter[class_folder] < 200:  # Her sınıf için 200 görüntüyü kullan
                image_path = os.path.join(class_path, image_file)
                if image_path.endswith(".jpg") or image_path.endswith(".png"):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (125, 100))
                    images.append(image)
                    labels.append(class_folder)
                    image_counter[class_folder] += 1

# Karıştırma ve eğitim/test setlerine bölme
data = list(zip(images, labels))
#np.random.shuffle(data)
images, labels = zip(*data)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# SIFT özniteliklerini çıkarma
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        descriptors = descriptors[:128, :].astype(np.float32)  # Veri tipini float32'ye dönüştürduk
        if descriptors.shape[0] < 128:
            zero_padding = np.zeros((128 - descriptors.shape[0], descriptors.shape[1]), dtype=np.float32)
            descriptors = np.vstack([descriptors, zero_padding])
    else:
        descriptors = np.zeros((128, 128), dtype=np.float32)
    return descriptors

# build_bow fonksiyonunu güncelledik
def build_bow(train_descriptors, num_clusters):
    all_descriptors = [descriptor for descriptor in train_descriptors if descriptor is not None]
    max_length = max(len(descriptor) for descriptor in all_descriptors)
    all_descriptors = [np.concatenate(descriptor, np.zeros((max_length - len(descriptor), descriptor.shape[1]), dtype=np.float32)) 
                        if len(descriptor) < max_length else descriptor for descriptor in all_descriptors]
    all_descriptors = np.vstack(all_descriptors)
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(all_descriptors)
    return kmeans

X_train_sift = [extract_sift_features(img) for img in train_images]
num_clusters = 100
kmeans = build_bow(X_train_sift, num_clusters)

# Görüntüleri temsil etmek için özellik vektörlerini oluşturduk
def represent_images(images, kmeans, num_clusters):
    features = []
    for img in images:
        sift_features = extract_sift_features(img)
        if sift_features is not None:
            img_bow = kmeans.predict(sift_features)
            hist, _ = np.histogram(img_bow, bins=num_clusters, range=(0, num_clusters))
            features.append(hist)
        else:
            features.append(np.zeros(num_clusters))
    return np.array(features)

X_train_bow = represent_images(train_images, kmeans, num_clusters)
X_test_bow = represent_images(test_images, kmeans, num_clusters)

# Standart ölçeklendirme uyguladik
scaler = StandardScaler()
X_train_bow_scaled = scaler.fit_transform(X_train_bow)
X_test_bow_scaled = scaler.transform(X_test_bow)

# k-NN modelini eğittik
knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(X_train_bow_scaled, train_labels)

# Test verileri üzerinde sınıflandırma yaptik
y_pred_knn = knn_model.predict(X_test_bow_scaled)

# SVM modeli
svm_model = SVC()
svm_model.fit(X_train_bow_scaled, train_labels)

# Rastgele Orman modeli
rf_model = RandomForestClassifier()
rf_model.fit(X_train_bow_scaled, train_labels)

# Test verileri üzerinde sınıflandırma yap ve doğruluk değerini hesapladik
y_pred_svm = svm_model.predict(X_test_bow_scaled)
accuracy_svm = accuracy_score(test_labels, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

y_pred_rf = rf_model.predict(X_test_bow_scaled)
accuracy_rf = accuracy_score(test_labels, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

# k-NN modelinin doğruluk değerini hesapladik
accuracy_knn = accuracy_score(test_labels, y_pred_knn)
print(f"k-NN Accuracy: {accuracy_knn}")

from sklearn.metrics import confusion_matrix

# k-NN için karmaşıklık matrisi
conf_matrix_knn = confusion_matrix(test_labels, y_pred_knn)
print("k-NN Karmaşıklık Matrisi:")
print(conf_matrix_knn)

# SVM için karmaşıklık matrisi
conf_matrix_svm = confusion_matrix(test_labels, y_pred_svm)
print("SVM Karmaşıklık Matrisi:")
print(conf_matrix_svm)

# Random Forest için karmaşıklık matrisi
conf_matrix_rf = confusion_matrix(test_labels, y_pred_rf)
print("Random Forest Karmaşıklık Matrisi:")
print(conf_matrix_rf)
