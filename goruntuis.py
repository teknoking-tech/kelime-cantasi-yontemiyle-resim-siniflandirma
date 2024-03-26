import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Veri seti dizini
dataset_dir = "C:/Users/FURKAN/Desktop/BitirmeProjem/data"

# Veri ve etiket listelerini tanımla
data = []
labels = []

# Veri seti dizinindeki sınıfları dön
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)


    # Sınıfın altındaki alt dizinlere dön
    for subdir in os.listdir(class_dir):
        subdir_path = os.path.join(class_dir, subdir)
   

        # Her sınıfın altındaki resimleri dön
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)

            # Resmi oku ve veri listesine ekle
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı olarak oku
            
            # Görüntü başarılı bir şekilde yüklenirse devam et
        
# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# SIFT özniteliklerini çıkarma
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# SIFT özniteliklerini eğitim verisi üzerinden çıkar
X_train_sift = [extract_sift_features(img) for img in X_train]

# K-means kümeleme ile kelime çantasını oluştur
def build_bow(train_descriptors, num_clusters):
    all_descriptors = np.vstack(train_descriptors)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_descriptors)
    return kmeans

num_clusters = 100  # Bu değeri ihtiyacınıza göre ayarlayabilirsiniz
kmeans = build_bow(X_train_sift, num_clusters)

# Görüntüleri temsil etmek için özellik vektörlerini oluştur
def represent_images(images, kmeans, num_clusters):
    features = []
    for img in images:
        sift_features = extract_sift_features(img)
        if sift_features is not None:
            img_bow = kmeans.predict(sift_features)
            hist, _ = np.histogram(img_bow, bins=num_clusters, range=(0, num_clusters))
            features.append(hist)
        else:
            features.append(np.zeros(num_clusters))  # Eğer SIFT öznitelikleri yoksa sıfır vektör ekle
    return np.array(features)

# Eğitim ve test verilerini temsil et
X_train_bow = represent_images(X_train, kmeans, num_clusters)
X_test_bow = represent_images(X_test, kmeans, num_clusters)

# Standart ölçeklendirme uygula
scaler = StandardScaler()
X_train_bow_scaled = scaler.fit_transform(X_train_bow)
X_test_bow_scaled = scaler.transform(X_test_bow)

# k-NN modelini eğit
knn_model = KNeighborsClassifier(n_neighbors=5)  # 5 komşu kullanılarak bir örnek
knn_model.fit(X_train_bow_scaled, y_train)

# Test verileri üzerinde sınıflandırma yap
y_pred = knn_model.predict(X_test_bow_scaled)

# Doğruluk değerini hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
