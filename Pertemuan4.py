#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi 2800 Data Audio Torronto

# Nama : Qoid Rif'at
# 
# NIM : 210411100160
# 
# Kelas : Proyek Sains Data (A)

# In[ ]:


pip install librosa


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/psd a/Dataset/TESS')


# In[ ]:


# Import Library
import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode


# In[ ]:


folders=['YAF_sad','YAF_pleasant_surprised','YAF_neutral',
         'YAF_happy','YAF_fear','YAF_disgust','YAF_angry',
         'OAF_Sad','OAF_Pleasant_surprise','OAF_neutral',
         'OAF_happy','OAF_Fear','OAF_disgust',
         'OAF_angry',
         ]


# In[ ]:


def calculate_statistics(audio_path):
    y, sr = librosa.load(audio_path)

    # UNTUK MENGHITUNG NILAI STATISTIKA
    mean = np.mean(y)
    std_dev = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    median = np.median(y)
    skewness = skew(y)  # Calculate skewness
    kurt = kurtosis(y)  # Calculate kurtosis
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    mode_value, _ = mode(y)  # Calculate mode
    iqr = q3 - q1

    # UNTUK MENGHITUNG NILAI ZCR
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
    zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
    zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
    zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])

    # UNTUK MENGHITUNG NILAI RMSE
    rmse = np.sum(y**2) / len(y)
    rmse_median = np.median(y**2)
    rmse_std_dev = np.std(y**2)
    rmse_kurtosis = kurtosis(y**2)
    rmse_skew = skew(y**2)

    return [zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew, rmse, rmse_median, rmse_std_dev, rmse_kurtosis, rmse_skew]


# In[ ]:


features =[]


# In[ ]:


for folder in folders:
    folder_path = f'{folder}'
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            audio_path = os.path.join(folder_path, filename)
            statistics = calculate_statistics(audio_path)
            features.append([folder, filename] + statistics)


# In[ ]:


# Membuat DataFrame dari data
columns =  ['Label', 'File'] + ['ZCR Mean', 'ZCR Median', 'ZCR Std Dev', 'ZCR Kurtosis', 'ZCR Skew', 'RMSE', 'RMSE Median', 'RMSE Std Dev', 'RMSE Kurtosis', 'RMSE Skew']
df = pd.DataFrame(features, columns=columns)
df


# In[ ]:


from google.colab import files
df.to_csv('pert4.csv', index=False)
display(df)

files.download('pert4.csv')


# ---
# **PRE-PROCESSING NORMALISASI dengan Z-SCORE**
# 
# ---
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
# Baca file CSV
dn = pd.read_csv("pert4.csv")
# Tentukan kolom yang akan distandarisasi
kolom = ['ZCR Mean', 'ZCR Median', 'ZCR Std Dev', 'ZCR Kurtosis', 'ZCR Skew', 'RMSE', 'RMSE Median', 'RMSE Std Dev', 'RMSE Kurtosis', 'RMSE Skew']
# Inisialisasi StandardScaler
scaler = StandardScaler()
# Lakukan standarisasi pada kolom yang telah ditentukan
dn[kolom] = scaler.fit_transform(dn[kolom])
# Simpan DataFrame yang telah distandarisasi ke dalam file CSV baru
dn.to_csv("pert4_normalisasi.csv", index=False)


# In[ ]:


norm=pd.read_csv('pert4_normalisasi.csv')
norm


# In[ ]:


# Daftar kolom yang ingin dilewati
kolomlabel= ['Label','File']
# Menghitung rata-rata untuk kolom numerik tertentu (mengabaikan kolom yang tidak diinginkan)
rata2= norm.drop(columns=kolomlabel).mean()
#membulatkan hasil komputasi dengan round dengan ketentuan 2 setelah koma, biar ga panjang bestiiiiiii
dibulatkan=rata2.round(2)
# Menampilkan rata-rata
print('--MEAN--')
print(dibulatkan)


# In[ ]:


# Daftar kolom yang ingin dilewati
kolomlabel= ['Label','File']
# Menghitung rata-rata untuk kolom numerik tertentu (mengabaikan kolom yang tidak diinginkan)
standv= norm.drop(columns=kolomlabel).std()
#membulatkan hasil komputasi dengan round dengan ketentuan 2 setelah koma, biar ga panjang bestiiiiiii
bulatkan=standv.round(2)
# Menampilkan rata-rata
print('--STANDARD DEVIASI--')
print(bulatkan)


# ---
# **NORMALISASI SETELAH SPLIT DATA DAN MENYIMPAN NORMALISASI DALAM BENTUK MODEL**
# 
# ---
# 
# Normalisasi bisa dilakukan sebelum atau sesudah split data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump


# In[ ]:


# Baca data dari file CSV
dataknn= pd.read_csv('pert4.csv')
# Pisahkan fitur (X) dan label (y)
X = dataknn.drop(['Label','File'], axis=1)  # Ganti 'target_column' dengan nama kolom target
y = dataknn['Label']
# split data into train and test sets
X_train,X_test,y_train, y_test= train_test_split(X, y, random_state=1, test_size=0.2)
# define scaler
scaler = StandardScaler()
# fit scaler on the training dataset
scaler.fit(X_train)
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))
# transform the training dataset
X_train_scaled = scaler.transform(X_train)


# In[ ]:


import pickle
with open('scaler.pkl', 'rb') as standarisasi:
    loadscal= pickle.load(standarisasi)


# In[ ]:


#normalisasi X testing dari hasil normalisasi X train yang disimpan dalam model
X_test_scaled=loadscal.transform(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


K = 30
acc = np.zeros((K-1))

for n in range(1,K,2):
    knn = KNeighborsClassifier(n_neighbors= n, metric = "euclidean").fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    acc[n-1]= accuracy_score(y_test,y_pred)

print('Akurasi terbaik adalah ', acc.max(), 'dengan nilai k =', acc.argmax()+1)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors= 13, metric = "euclidean")
dump(knn, open('modelknn.pkl', 'wb'))


# In[ ]:


import pickle
with open('modelknn.pkl', 'rb') as knn:
    loadknn= pickle.load(knn)
loadknn.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = loadknn.predict(X_test_scaled)
y_pred


# In[ ]:


accuracy = accuracy_score(y_test,y_pred)
print("Akurasi:",accuracy)


# In[ ]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=8)
X_train_pca= sklearn_pca.fit_transform(X_train_scaled)
type(X_train_pca)


# In[ ]:


dump(sklearn_pca, open('PCA8.pkl', 'wb'))


# In[ ]:


import pickle
with open('PCA8.pkl', 'rb') as pca:
    loadpca= pickle.load(pca)

X_test_pca=loadpca.transform(X_test_scaled)
X_test_pca.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train_pca, y_train)


# In[ ]:


y_prediksi = classifier.predict(X_test_pca)
y_prediksi


# In[ ]:


acc_pca= accuracy_score(y_test,y_prediksi)
print("Akurasi:",acc_pca)
#Akurasi: 0.7375


# In[ ]:


import numpy as np
import matplotlib .pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Menggunakan data yang sudah Anda miliki
X_latih = X_train_scaled
X_uji = X_test_scaled

# Membuat pipeline
pipeline = Pipeline([
    ('analisa_komponen_utama', sklearnPCA()),
    ('k_terdekat', KNeighborsClassifier(metric='euclidean'))
])

# Menentukan parameter grid
param_grid = {
    'analisa_komponen_utama__n_components': [i for i in range(1, X_latih.shape[1] + 1)],  # Perbaiki rentang berdasarkan jumlah fitur
    'k_terdekat__n_neighbors':list(range(1, 51)),
    'k_terdekat__weights': ['uniform', 'distance']
}

# Melakukan GridSearch
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_latih, y_train)  # Asumsikan y_latih sudah ada

# Hasil dari GridSearch
hasil_pencarian = grid_search.cv_results_

# Ambil nilai rata-rata skor tes
skor = hasil_pencarian['mean_test_score']

# Membuat heatmap dari skor
matriks_skor = np.array(skor).reshape(len(param_grid['k_terdekat__n_neighbors']),
                                      len(param_grid['analisa_komponen_utama__n_components']),
                                      len(param_grid['k_terdekat__weights']))

# Visualisasi untuk bobot 'seragam'/pendekatan uniform
sns.heatmap(matriks_skor[:, :, 0], annot=True, cmap='viridis',
            xticklabels=param_grid['analisa_komponen_utama__n_components'],
            yticklabels=param_grid['k_terdekat__n_neighbors'])
plt.title('Peta Akurasi (Bobot: Seragam/UNIFORM)')
plt.xlabel('Jumlah Komponen PCA')
plt.ylabel('Jumlah Tetangga')
plt.show()

# Visualisasi untuk bobot 'jarak' / distance
sns.heatmap(matriks_skor[:, :, 1], annot=True, cmap='viridis',
            xticklabels=param_grid['analisa_komponen_utama__n_components'],
            yticklabels=param_grid['k_terdekat__n_neighbors'])
plt.title('Peta Akurasi (Bobot: Jarak/DISTANCE)')
plt.xlabel('Jumlah Komponen PCA')
plt.ylabel('Jumlah Tetangga')
plt.show()

