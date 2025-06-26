
#11. To implement a Next Word Prediction Using an RNN on Simple English Sentences

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Sample sentences
sentences = ["I like to eat", "I like to play", "She loves pizza"]

# 2. Tokenize words
tok = Tokenizer()
tok.fit_on_texts(sentences)

# 3. Create training sequences
seqs = []
for s in sentences:
    tokens = tok.texts_to_sequences([s])[0]
    for i in range(1, len(tokens)):
        seqs.append(tokens[:i+1])

# 4. Pad sequences
max_len = max(len(x) for x in seqs)
seqs = pad_sequences(seqs, maxlen=max_len)

# 5. Inputs and labels
X = seqs[:, :-1]
y = tf.keras.utils.to_categorical(seqs[:, -1], num_classes=len(tok.word_index)+1)

# 6. Build RNN model
model = Sequential([
    Embedding(len(tok.word_index)+1, 8, input_length=max_len-1),
    SimpleRNN(16),
    Dense(len(tok.word_index)+1, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=200, verbose=0)

# 7. Predict next word
def predict(text):
    seq = tok.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len-1)
    pred = np.argmax(model.predict(seq, verbose=0))
    for word, idx in tok.word_index.items():
        if idx == pred:
            return text + " " + word

# 8. Try it
print(predict("I like to"))
print(predict("She loves"))
