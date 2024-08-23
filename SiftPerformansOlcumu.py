# Kütüphanelerin import edilmesi
import cv2  # OpenCV kütüphanesini kullanmak için
import numpy as np  # NumPy kütüphanesini kullanmak için
from sklearn.cluster import KMeans  # KMeans kümeleme algoritması için
from sklearn.neighbors import KNeighborsClassifier  # k-nn sınıflandırma algoritması için
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Sınıflandırma performans ölçütleri için
from sklearn.model_selection import train_test_split  # Veri setini bölme için
import os  # İşletim sistemi fonksiyonları için
import matplotlib.pyplot as plt  # Veri görselleştirme için
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog


# Adım 1: Eğitim ve test veri setlerinin oluşturulması
train_dir = "train verisinin yolu"  # Eğitim veri seti dizini
test_dir = "test verisinin yolu"  # Test veri seti dizini

# Veri setini yükleyen fonksiyon
def load_data(directory):
    images = []  # Görüntülerin depolanacağı liste
    labels = []  # Etiketlerin depolanacağı liste
    class_names = os.listdir(directory)  # Sınıf isimlerini al
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)  # Sınıfın dizinini oluştur
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)  # Görüntünün dosya yolunu oluştur
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamalı olarak oku
            
            # Adım 2: Ön işlem
            # Histogram Eşitleme
            img = cv2.equalizeHist(img)
            
            # Gürültü Azaltma
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Boyut Standardizasyonu
            img = cv2.resize(img, (256, 256))
            
            images.append(img)
            labels.append(class_name)
    return images, labels

# Eğitim ve test veri setlerini yükle
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# Adım 3: SIFT özniteliklerinin çıkarılması
def extract_sift_features(image):
    sift = cv2.SIFT_create()  # SIFT öznitelik çıkarıcı nesnesi oluştur
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Görüntüden öznitelikleri çıkar
    return descriptors

# Eğitim veri setindeki her bir görüntü için SIFT özniteliklerini çıkar
train_descriptors = [extract_sift_features(img) for img in train_images]

# Adım 4: Kelime Sözlüğünün oluşturulması
all_descriptors = np.concatenate(train_descriptors, axis=0)  # Tüm öznitelikleri birleştir
k_clusters = 100  # Kümelerin sayısını isteğe bağlı olarak ayarla
kmeans = KMeans(n_clusters=k_clusters, n_init=10)  # KMeans kümeleme algoritması nesnesi oluştur
kmeans.fit(all_descriptors)  # Kelime sözlüğünü oluştur
visual_words = kmeans.cluster_centers_

# Adım 5: Test görüntülerinin özniteliklerinin elde edilmesi
def extract_image_features(image, visual_words):
    sift = cv2.SIFT_create()  # SIFT öznitelik çıkarıcı nesnesi oluştur
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Görüntüden öznitelikleri çıkar
    features = np.zeros(len(visual_words))

    for descriptor in descriptors:
        distances = np.linalg.norm(visual_words - descriptor, axis=1)  # Uzaklıkları hesapla
        min_index = np.argmin(distances)  # En küçük uzaklık indeksini bul
        features[min_index] += 1  # Histogramı güncelle

    return features / np.sum(features)  # Histogramı normalize et

# Test veri setindeki her bir görüntü için öznitelikleri çıkar
test_features = [extract_image_features(img, visual_words) for img in test_images]

# Adım 6: Test görüntülerinin sınıflandırılması
def classify_images(train_features, train_labels, test_features, k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)  # k-nn sınıflandırıcı nesnesi oluştur
    knn.fit(train_features, train_labels)  # Modeli eğit
    predicted_labels = knn.predict(test_features)  # Test verilerini sınıflandır
    return predicted_labels

# Adım 7: Performans sonuçları
k_neighbors = 5  # k-nn için komşu sayısını isteğe bağlı olarak ayarlayabilirsiniz

# Eğitim veri setindeki her bir görüntü için öznitelikleri çıkar
train_features = [extract_image_features(img, visual_words) for img in train_images]
# Test görüntülerini sınıflandır
predicted_labels = classify_images(train_features, train_labels, test_features, k_neighbors)

# Doğruluk, karmaşıklık matrisi ve sınıflandırma raporu hesapla
accuracy = accuracy_score(test_labels, predicted_labels)
confusion_mat = confusion_matrix(test_labels, predicted_labels)
class_report = classification_report(test_labels, predicted_labels, zero_division=1)

# Sonuçları yazdır
print("DOGRULUK:", accuracy * 100)
print("KARISIKLIK MATRISI:\n", confusion_mat)
print("\nSINIFLANDIRMA SONUCU:\n", class_report)

# Adım 8: Matplotlib ile öznitelik histogramlarını göster
# Sınıf isimlerini sırala
class_names = sorted(list(set(test_labels)))

# Her sınıf için öznitelik histogramlarını göster
for i, class_name in enumerate(class_names):
    class_indices = [j for j, label in enumerate(test_labels) if label == class_name]
    class_features = np.array([test_features[j] for j in class_indices])
    class_mean_features = np.mean(class_features, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_mean_features)), class_mean_features, color='blue', alpha=0.7)
    plt.title(f"SINIF: {class_name}")
    plt.xlabel("GORSEL KELIME DIZINI")
    plt.ylabel("NORMALLESTIRILMIS FREKANS ")
    plt.show()


# HOG  öznitelik cıkarma fonksiyonu
def extract_hog_features(images):
    feature_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, transform_sqrt=True)
        feature_list.append(hog_features)
    return np.array(feature_list)
# Color Histogram fonksiyonu
def extract_color_histogram(images):
    feature_list = []
    for img in images:
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feature_list.append(hist)
    return np.array(feature_list)
# kenar histogram fonksiyonu
def edge_histogram(images):
    feature_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        hist, _ = np.histogram(edges, bins=256, range=(0, 256))
        feature_list.append(hist)
    return np.array(feature_list)

def train_and_evaluate_models(data_folder):
    images = []
    labels = []

    for class_folder in ["train", "test"]:
        class_path = os.path.join(data_folder, class_folder)
        for class_name in os.listdir(class_path):
            class_images = []
            class_labels = []
            class_label = class_name

            class_folder_path = os.path.join(class_path, class_name)
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                if image_path.endswith((".jpg", ".png")):
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (200, 100))
                    class_images.append(img)
                    class_labels.append(class_label)

            images.extend(class_images)
            labels.extend(class_labels)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # HOG ile öznitelik çıkarma
    knn_hog = KNeighborsClassifier()
    svm_hog = SVC()
    rf_hog = RandomForestClassifier()

    train_features_hog = extract_hog_features(train_images)
    test_features_hog = extract_hog_features(test_images)

    knn_hog.fit(train_features_hog, train_labels)
    svm_hog.fit(train_features_hog, train_labels)
    rf_hog.fit(train_features_hog, train_labels)

    knn_pred_hog = knn_hog.predict(test_features_hog)
    svm_pred_hog = svm_hog.predict(test_features_hog)
    rf_pred_hog = rf_hog.predict(test_features_hog)

    knn_accuracy_hog = accuracy_score(test_labels, knn_pred_hog)
    svm_accuracy_hog = accuracy_score(test_labels, svm_pred_hog)
    rf_accuracy_hog = accuracy_score(test_labels, rf_pred_hog)

    print("HOG ile K-En Yakın Komşular Doğruluk: {:.2f}".format(knn_accuracy_hog))
    print("HOG ile Destek Vektör Makineleri Doğruluk: {:.2f}".format(svm_accuracy_hog))
    print("HOG ile Rastgele Orman Doğruluk: {:.2f}".format(rf_accuracy_hog))

    # Color histogram ile öznitelik çıkarma
    knn_color = KNeighborsClassifier()
    svm_color = SVC()
    rf_color = RandomForestClassifier()

    train_features_color = extract_color_histogram(train_images)
    test_features_color = extract_color_histogram(test_images)

    knn_color.fit(train_features_color, train_labels)
    svm_color.fit(train_features_color, train_labels)
    rf_color.fit(train_features_color, train_labels)

    knn_pred_color = knn_color.predict(test_features_color)
    svm_pred_color = svm_color.predict(test_features_color)
    rf_pred_color = rf_color.predict(test_features_color)

    knn_accuracy_color = accuracy_score(test_labels, knn_pred_color)
    svm_accuracy_color = accuracy_score(test_labels, svm_pred_color)
    rf_accuracy_color = accuracy_score(test_labels, rf_pred_color)

    print("Color Histogram ile K-En Yakın Komşular Doğruluk: {:.2f}".format(knn_accuracy_color))
    print("Color Histogram ile Destek Vektör Makineleri Doğruluk: {:.2f}".format(svm_accuracy_color))
    print("Color Histogram ile Rastgele Orman Doğruluk: {:.2f}".format(rf_accuracy_color))

    # Edge histogram ile öznitekil çıkarma
    knn_edge = KNeighborsClassifier()
    svm_edge = SVC()
    rf_edge = RandomForestClassifier()

    train_features_edge = edge_histogram(train_images)
    test_features_edge = edge_histogram(test_images)

    knn_edge.fit(train_features_edge, train_labels)
    svm_edge.fit(train_features_edge, train_labels)
    rf_edge.fit(train_features_edge, train_labels)

    knn_pred_edge = knn_edge.predict(test_features_edge)
    svm_pred_edge = svm_edge.predict(test_features_edge)
    rf_pred_edge = rf_edge.predict(test_features_edge)

    knn_accuracy_edge = accuracy_score(test_labels, knn_pred_edge)
    svm_accuracy_edge = accuracy_score(test_labels, svm_pred_edge)
    rf_accuracy_edge = accuracy_score(test_labels, rf_pred_edge)

    print("Edge Histogram ile K-En Yakın Komşular Doğruluk: {:.2f}".format(knn_accuracy_edge))
    print("Edge Histogram ile Destek Vektör Makineleri Doğruluk: {:.2f}".format(svm_accuracy_edge))
    print("Edge Histogram ile Rastgele Orman Doğruluk: {:.2f}".format(rf_accuracy_edge))

# Fonksiyonu çağırma
train_and_evaluate_models("data verilerin olduğu dosyanın yolu")