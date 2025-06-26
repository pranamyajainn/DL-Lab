# Q12 To design and implement a customer support chatbot that leverages Bidirectional LSTMs

import tensorflow as tf
import numpy as np

# Training data: messages and their types (intents)
texts = ["hi", "hello", "what's up", "how are you",
         "tell me about you", "who are you", "what can you do",
         "bye", "goodbye"]

labels = ["greeting", "greeting", "greeting", "greeting",
          "info", "info", "info",
          "farewell", "farewell"]

# Convert labels to numbers
label_to_num = {"greeting": 0, "info": 1, "farewell": 2}
num_to_label = {0: "greeting", 1: "info", 2: "farewell"}
y = [label_to_num[label] for label in labels]

# Convert text to numbers
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = max(len(seq) for seq in sequences)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=8, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile and train (only 3 epochs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded, np.array(y), epochs=3, verbose=1)

# Bot reply function
def chatbot(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded_seq, verbose=0)
    intent = num_to_label[np.argmax(pred)]

    if intent == "greeting":
        return "Hello! 😊"
    elif intent == "info":
        return "I'm a small chatbot that understands a few things."
    elif intent == "farewell":
        return "Goodbye! 👋"
    else:
        return "Hmm... I didn't understand that."

# Chat loop
print("Chatbot ready! Type 'quit' to stop.")
while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        print("Bot: Bye!")
        break
    print("Bot:", chatbot(msg))
