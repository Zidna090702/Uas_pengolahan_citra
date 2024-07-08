import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar asli dan gambar tersegmentasi berdampingan
def show_images(original, segmented):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(segmented)
    ax[1].set_title(f'Segmented Image {k}')
    ax[1].axis('off')
    
    plt.show()

# Membaca gambar
image_path = 'img/sunflowerthai.jpg'
image = cv2.imread(image_path)

# Verifikasi apakah gambar berhasil dibaca
if image is None:
    print(f"Error: Tidak dapat membuka atau membaca file gambar di '{image_path}'")
    exit()

# Menyimpan gambar asli untuk ditampilkan nanti
original_image = image.copy()

(h, w) = image.shape[:2]

# Mengubah gambar ke dalam bentuk data array
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))

# Menentukan jumlah cluster k
k = 6

# Inisialisasi KMeans
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(image)

# Mendapatkan label dan pusat cluster
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Mengubah setiap piksel ke warna pusat cluster
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((h, w, 3))
segmented_image = segmented_image.astype('uint8')

# Menampilkan hasil segmentasi berdampingan dengan gambar asli
show_images(original_image, segmented_image)

# Menyimpan hasil segmentasi
cv2.imwrite('segmented_image.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
