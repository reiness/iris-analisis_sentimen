

![IRIS PROJECT](pro@4x-1.png)

# **Analisis Sentimen**

## **Deskripsi Proyek**
Proyek ini bertujuan untuk menganalisis sentimen dari deskripsi berita menggunakan model LSTM (Long Short-Term Memory). Data yang digunakan diambil dari file CSV yang berisi deskripsi dan label sentimen. Model ini menggunakan teknik ekstraksi fitur TF-IDF untuk mengubah teks menjadi representasi numerik.

## **Struktur Proyek**

### 1. Import Library
Mengimpor library yang diperlukan untuk analisis data, pemrosesan teks, dan pembangunan model.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
```

### 2. Download Stopwords
Mengunduh stopwords yang diperlukan dari NLTK untuk pemrosesan teks.

```python
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Load Data
Memuat data dari file CSV dan menampilkan beberapa baris pertama dari dataset.

```python
df = pd.read_csv('reliance_news_sentiment.csv')
df = df[['description', 'sentiment']]
df.head()
```

### 4. Visualisasi Data
Membuat plot untuk menunjukkan distribusi sentimen dalam dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns

cnt_pro = df['sentiment'].value_counts()
plt.figure(figsize=(12, 4))
sns.barplot(x=cnt_pro.index, y=cnt_pro.values, palette="viridis", alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Sentiment', fontsize=12)
plt.xticks(rotation=90)
plt.show()
```

### 5. Preprocessing Teks
Fungsi untuk melakukan preprocessing pada teks, termasuk konversi ke huruf kecil, penghapusan karakter non-alfabet, penghapusan stopwords, dan stemming.

```python
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('English'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = word_tokenize(text)  
    tokens = [token for token in tokens if token not in stop_words]  
    tokens = [stemmer.stem(token) for token in tokens] 
    return ' '.join(tokens)  

df['description'] = df['description'].apply(preprocess_text)

label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
```

### 6. Split Data
Membagi data menjadi set pelatihan dan pengujian.

```python
X = df['description'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. Ekstraksi Fitur
Menggunakan TF-IDF untuk mengekstrak fitur dari teks.

```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
```

### 8. Terapkan SMOTE
Menggunakan SMOTE untuk mengatasi ketidakseimbangan kelas pada data pelatihan.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)
```

### 9. Membangun Model LSTM
Membangun model LSTM untuk analisis sentimen.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

history = model.fit(
    X_train_tfidf, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_tfidf, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

### 10. Evaluasi Model
Menilai kinerja model menggunakan data pengujian.

```python
loss, accuracy = model.evaluate(X_test_tfidf, y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')

y_pred = model.predict(X_test_tfidf)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_classes))
```

### 11. Confusion Matrix
Menghitung dan menampilkan confusion matrix.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_classes)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### 12. Visualisasi Hasil
Membuat plot untuk loss dan akurasi selama pelatihan.

```python
# Plot loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
```

### 13. Simpan Model
Menyimpan model terlatih untuk digunakan di masa depan.

```python
model.save('sentiment_analysis_lstm_model.h5')
```

### 14. Prediksi dengan Model yang Disimpan
Memuat model yang disimpan dan melakukan prediksi pada deskripsi baru.

```python
from tensorflow.keras.models import load_model

loaded_model = load_model('sentiment_analysis_lstm_model.h5')

def preprocess_new_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

new_descriptions = ["I love this product!", "The service was terrible."]
new_descriptions_processed = [preprocess_new_text(desc) for desc in new_descriptions]

new_tfidf = tfidf_vectorizer.transform(new_descriptions_processed).toarray()
predictions = loaded_model.predict(new_tfidf)
predicted_classes = np.argmax(predictions, axis=1)

for desc, pred in zip(new_descriptions, predicted_classes):
    sentiment = label_encoder.inverse_transform([pred])[0]
    print(f"Description: '{desc}' -> Predicted Sentiment: '{sentiment}'")
```

## **Kesimpulan**
Proyek ini menunjukkan bagaimana melakukan analisis sentimen menggunakan model LSTM dan teknik pemrosesan teks seperti TF-IDF dan SMOTE untuk menangani ketidakseimbangan kelas. Dengan memanfaatkan framework Keras dan imbalanced-learn, kita dapat membangun model yang dapat memprediksi sentimen dengan akurasi yang baik.
