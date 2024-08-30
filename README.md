![IRIS PROJECT](pro@4x-1.png)

<h1 align="center">Analisis Sentimen</h1>

```markdown
# IRIS Sentiment Analysis Project

## Overview
This project implements a sentiment analysis model using Word2Vec embeddings and LSTM neural networks. The model is trained on a dataset of text samples with sentiment labels (positive, negative, neutral) and can predict the sentiment of new text inputs.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Word2Vec Model](#word2vec-model)
3. [Text Preprocessing](#text-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

## Data Preparation

The dataset is loaded from a CSV file containing text samples and their corresponding sentiment labels. Initial data exploration and visualization are performed to understand the distribution of sentiment labels.
```

```python
df = pd.read_csv('/content/sample_data/all-data.csv', encoding='latin-1', names=['Sentiment', 'Text'])
```

## Word2Vec Model

A Word2Vec model is trained on the preprocessed text data to generate word embeddings.

```python
model = gensim.models.Word2Vec(sentences=text_lines, vector_size=EMBEDDING_DIM, window=5, min_count=1)
```

The trained model is saved in ASCII format for later use:

```python
model.wv.save_word2vec_format('imdb_embedding_word2vec.txt', binary=False)
```

## Text Preprocessing

Text data is preprocessed using the following steps:
1. Tokenization
2. Conversion to lowercase
3. Removal of punctuation and non-alphabetic tokens
4. Removal of stop words

```python
def preprocess_inputs(df):
    # Preprocessing code here
```

## Model Architecture

The sentiment analysis model uses an LSTM architecture:

```python
model = Sequential()
model.add(Embedding(input_dim=9230, output_dim=100, input_length=100))
model.add(LSTM(32))
model.add(Dense(units=4, activation='softmax'))
```

## Training

The model is trained using the following parameters:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Batch size: 128
- Epochs: 25

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train_encoded, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test_encoded), verbose=2)
```

## Evaluation

The model achieves the following performance on the test set:
- Test accuracy: 71.41%

```python
score, acc = model.evaluate(X_test_pad, y_test_encoded, batch_size=128)
print('Test accuracy:', acc)
```

## Usage

To use the trained model for sentiment prediction:

1. Load the saved model architecture and weights:

```python
with open('model_architecture_Sentiment_classifier_word2vec_first_try.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('Sentiment_Classifier_word2vec_first_try.h5')
```

2. Preprocess new text data using the same steps as in training.
3. Use the model to predict sentiment:

```python
predictions = model.predict(new_text_pad)
```

## Dependencies

- numpy
- pandas
- nltk
- gensim
- tensorflow
- keras
- scikit-learn
- matplotlib
- seaborn

Install dependencies using:

```
pip install numpy pandas nltk gensim tensorflow keras scikit-learn matplotlib seaborn
```
