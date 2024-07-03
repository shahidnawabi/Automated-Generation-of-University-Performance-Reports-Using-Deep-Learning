import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Concatenate, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess your dataset
# Replace with your data loading and preprocessing code
text_data = [...]  # List of text samples
labels = [...]     # List of corresponding labels (e.g., 0 for class A, 1 for class B)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Tokenize and pad sequences
max_sequence_length = 128  # Define the maximum sequence length
vocab_size = 10000         # Define the vocabulary size

# Tokenize and pad sequences for BERT
bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_layer = hub.KerasLayer(bert_url, trainable=False)

tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int")
tokenizer.adapt(X_train)
X_train_seq = tokenizer(X_train)
X_test_seq = tokenizer(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# Define the model architecture
embedding_dim = 100  # Define the embedding dimension
cnn_filters = 128    # Number of filters in the CNN layers
lstm_units = 64      # Define the number of LSTM units
dropout_rate = 0.5   # Define the dropout rate

# Input layers
input_layer_bert = Input(shape=(max_sequence_length, 768))  # BERT embeddings shape
input_layer_cnn = Input(shape=(max_sequence_length,))

# BERT layer
bert_output = GlobalAveragePooling1D()(input_layer_bert)

# CNN layers
cnn_layer = Conv1D(filters=cnn_filters, kernel_size=3, activation='relu')(input_layer_cnn)
max_pooling_layer = MaxPooling1D()(cnn_layer)

# LSTM layer
bi_lstm = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer_cnn)

# Concatenate BERT, CNN, and LSTM outputs
concatenated_layer = Concatenate()([bert_output, max_pooling_layer, bi_lstm])

dropout_layer = Dropout(rate=dropout_rate)(concatenated_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

model = Model(inputs=[input_layer_bert, input_layer_cnn], outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Train the model
num_epochs = 5  # Define the number of training epochs
batch_size = 32  # Define the batch size
model.fit([X_train_bert, X_train_pad], y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
y_pred = model.predict([X_test_bert, X_test_pad])
y_pred_binary = (y_pred > 0.5).astype(int)

# Print classification report
print(classification_report(y_test, y_pred_binary))
