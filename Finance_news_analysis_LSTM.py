# -*- coding: utf-8 -*-
"""
# Import data
"""

import pandas as pd

df = pd.read_csv('reliance_news_sentiment.csv')

df.head()

df = df[['description', 'sentiment']]
df.head()

"""# **Prepo**

## mengecilkan semua huruf
"""

df['description'] = df['description'].str.lower()

"""## menghitung jumlah total kata dalam semua entri dalam kolom 'description' DataFrame."""

df.index = range(653)
df['description'].apply(lambda x: len(x.split(' '))).sum()

"""## Visualisasi"""

import matplotlib.pyplot as plt
import seaborn as sns

cnt_pro = df['sentiment'].value_counts()
plt.figure(figsize=(12, 4))
sns.barplot(x=cnt_pro.index, y=cnt_pro.values, palette="viridis", alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Sentiment', fontsize=12)
plt.xticks(rotation=90)
plt.show()

"""## mengubah class (neutral, positive, negative) menjadi numerik"""

#Convert sting to numeric
sentiment  = {'positive': 0,'neutral': 1,'negative':2}

df.sentiment = [sentiment[item] for item in df.sentiment]
df

def print_message(index):
    example = df[df.index == index][['description', 'sentiment']].values[0]
    if len(example) > 0:
        print(example[0])
        print('description:', example[1])
print_message(12)

print_message(0)

"""## membersihkan text"""

from bs4 import BeautifulSoup
import re

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.replace('x', '')
    return text
df['description'] = df['description'].apply(cleanText)

from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk

import nltk
nltk.download('punkt')

train, test = train_test_split(df, test_size=0.00002, random_state=42)

import nltk
from nltk.corpus import stopwords

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['description']), tags=[r.sentiment]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['description']), tags=[r.sentiment]), axis=1)

# The maximum number of words to be used. (most frequent)
max_fatures = 500000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['description'].values)
X = tokenizer.texts_to_sequences(df['description'].values)
X = pad_sequences(X)
print('Found %s unique tokens.' % len(X))

X = tokenizer.texts_to_sequences(df['description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#train_tagged.values[2173]
train_tagged.values

from gensim.models import Doc2Vec
from tqdm import tqdm

d2v_model = Doc2Vec(dm=1, dm_mean=1, vector_size=20, window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])

from sklearn import utils

# Commented out IPython magic to ensure Python compatibility.
# %%time
# for epoch in range(30):
#     d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
#     d2v_model.alpha -= 0.002
#     d2v_model.min_alpha = d2v_model.alpha

import numpy as np

# Save the vectors in a new matrix
embedding_matrix = np.zeros((len(d2v_model.wv.key_to_index)+ 1, 20))

for key, i in d2v_model.wv.key_to_index.items():
    embedding_vector = d2v_model.wv[key]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

d2v_model.wv.most_similar(positive=['profit'], topn=10)

d2v_model.wv.most_similar(positive=['investment'], topn=10)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding


# init layer
model = Sequential()

# emmbed word vectors
model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=X.shape[1], weights=[embedding_matrix], trainable=True))

# learn the correlations
def split_input(sequence):
     return sequence[:-1], tf.reshape(sequence[1:], (-1,1))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(3,activation="softmax"))

# output model skeleton
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

from keras.utils import plot_model
plot_model(model, to_file='model.png')

Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

X_train

batch_size = 32
history=model.fit(X_train, Y_train, epochs =50, batch_size=batch_size, verbose = 2)

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')

# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=2)
_, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
print(yhat_probs)

# predict classes based on probabilities
yhat_classes = np.argmax(yhat_probs, axis=1)
print(yhat_classes)

# The confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
rounded_labels=np.argmax(Y_test, axis=1)

lstm_val = confusion_matrix(rounded_labels, yhat_classes)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(lstm_val, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
plt.title('LSTM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

print(X_test.shape)

validation_size = 30

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)

print("acc: %.2f" % (acc))

"""# Save Model"""

model.save('Mymodel.h5')

"""#pakai model"""

note : sentiment = {'positive': 0,'neutral': 1,'negative':2}

message = ['Congratulations! you have won a $1,000 Walmart gift card']
seq = tokenizer.texts_to_sequences(message)

padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)

pred = model.predict(padded)

labels = ['0','1','2']
print(pred, labels[np.argmax(pred)])