import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessor import MyanmarTextPreprocessor

from preprocessor import MyanmarTextPreprocessor

# ---- Configuration ----
MAXLEN = 100
VOCAB_SIZE = 40701
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons, dropout_rate=0.5,**kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(neurons, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=False):
        # Multi-head self-attention with mask
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Token + Position Embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,**kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
map_label = {
    'Social': 0,
    'Entertainment': 1,
    'Product&Service': 2,
    'Business': 3,
    'Sports': 4,
    'Science&Technology': 5,
    'Education': 6,
    'Culture&History': 7,
    'Health': 8,
    'Environmental': 9,
    'Political': 10,
    'Gambling': 11,
    'Adult Content': 12,
}

# ---- Load tokenizer and model ----
@st.cache_resource
def load_assets():
    model = load_model('model_saveFile/my_transformer_model.h5', custom_objects={
        'TransformerEncoder': TransformerEncoder,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    })
    with open('model_saveFile/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer, model

tokenizer, model = load_assets()

# ---- Label map ----
map_label = {
    0: 'Social',
    1: 'Entertainment',
    2: 'Product&Service',
    3: 'Business',
    4: 'Sports',
    5: 'Science&Technology',
    6: 'Education',
    7: 'Culture&History',
    8: 'Health',
    9: 'Environmental',
    10: 'Political',
    11: 'Gambling',
    12: 'Adult Content'
}

# ---- Input UI ----
st.title("Myanmar Text Classification")
user_input = st.text_area("Enter a Myanmar sentence:")
dict_path = 'dict-words.txt'
sw_path = 'sw.txt'
preprocessor = MyanmarTextPreprocessor(dict_path, sw_path)
# ---- Predict ----
def predict(text):
    cleaned = preprocessor.preprocessing(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    seq = [[min(token, VOCAB_SIZE - 1) for token in s] for s in seq]  # clamp to avoid embedding error
    pad_seq = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')

    pred_probs = model.predict(pad_seq)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    pred_label = map_label.get(pred_index, "Unknown")
    confidence = pred_probs[0][pred_index]
    return pred_label, confidence

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        label, conf = predict(user_input)
        st.success(f"Predicted category: **{label}** with confidence **{conf:.2f}**")
