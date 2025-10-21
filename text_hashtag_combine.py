# -------------------- IMPORTS --------------------
import os
import sys
import logging
import streamlit as st
import pickle
import unicodedata
import numpy as np
import tensorflow as tf
import torch
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer, util
from preprocessor import MyanmarTextPreprocessor

# -------------------- ENVIRONMENT CONFIG --------------------
# Suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------- CUSTOM LAYERS --------------------
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons, dropout_rate=0.5, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(neurons, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=False):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_output_shape(self, input_shape):
        return input_shape + (self.token_emb.output_dim,)



# -------------------- LOAD RESOURCES --------------------
# Paths
MODEL_PATH = 'my_transformer_model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
DICT_PATH = 'dict-output-v2-4-9-2025.txt'
SW_PATH = 'sw.txt'
BAN_WORDS_PATH = "ban-words-4-9-2025.txt"
CATEGORY_CSV = 'Distinct-Words-21-8-2025.csv'

# Label mapping
map_label = {
    'Social': 0, 'Entertainment': 1, 'Product&Service': 2, 'Business': 3, 'Sports': 4,
    'Science&Technology': 5, 'Education': 6, 'Culture&History': 7, 'Health': 8,
    'Environmental': 9, 'Political': 10, 'Gambling': 11, 'Adult Content': 12,
}

maxlen = 100
vocab_limit = 40701

# Load model safely
try:
    model = load_model(MODEL_PATH, custom_objects={
        'TransformerEncoder': TransformerEncoder,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    })
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load tokenizer safely
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# Preprocessor
preprocessor = MyanmarTextPreprocessor(DICT_PATH, SW_PATH)

# Load category words
def load_category_words_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    category_words = {}
    for col in df.columns:
        words = df[col].dropna().astype(str).tolist()
        category_words[col.strip()] = set(w.strip() for w in words if w.strip())
    return category_words

category_words = load_category_words_from_csv(CATEGORY_CSV)

# -------------------- HELPER FUNCTIONS --------------------
def normalize_font_style(text):
    return unicodedata.normalize('NFKC', text)

def safe_pad_sequences(texts, tokenizer, maxlen, vocab_limit):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = [[min(token, vocab_limit - 1) for token in seq] for seq in sequences]
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

def predict_label_from_tokens(tokens, category_words):
    score_dict = {}
    for label, keywords in category_words.items():
        matches = set(tokens) & keywords
        score_dict[label] = len(matches)
    predicted_label = max(score_dict, key=score_dict.get)
    return predicted_label, score_dict

def essential(path, tokens):
    with open(path, 'r', encoding="utf-8") as f:
        temp_ban = set(line.strip().lower() for line in f if line.strip())
    filtered_tokens = [token for token in tokens if token in temp_ban]
    return filtered_tokens

# -------------------- SENTENCE TRANSFORMER --------------------
try:
    st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    st.error(f"SentenceTransformer load failed: {e}")
    st_model = None

st_categories = [
    'Social', 'Entertainment', 'Product', 'Service', 'Business', 'Sports',
    'Science', 'Technology', 'History', 'Education', 'Culture',
    'Health', 'Environmental'
]

if st_model:
    st_embeddings = st_model.encode(st_categories, convert_to_tensor=True)

def classify_text_semantically(text):
    if not st_model:
        return "Unknown", 0.0
    text = text.lstrip("#").lower()
    text_emb = st_model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(text_emb, st_embeddings)[0]
    best_idx = torch.argmax(sims).item()
    return st_categories[best_idx], sims[best_idx].item()

# -------------------- STREAMLIT UI --------------------
st.title("Myanmar Text Classification & Hashtag Analyzer")
input_text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        text = normalize_font_style(input_text).lower()

        # Hashtags
        hashtags = [word for word in text.split() if word.startswith("#")]
        hashtag_results = []
        for tag in hashtags:
            label, score = classify_text_semantically(tag)
            hashtag_results.append({
                "hashtag": tag,
                "semantic_label": label,
                "similarity_score": round(score, 4)
            })

        # Preprocessing + Transformer model
        cleaned = preprocessor.preprocessing(text)
        if not cleaned:
            cleaned = text
        if tokenizer and model:
            seq_pad = safe_pad_sequences([cleaned], tokenizer, maxlen, vocab_limit=vocab_limit)
            try:
                pred_probs = model.predict(seq_pad)
                pred_index = np.argmax(pred_probs, axis=1)[0]
                inv_map_label = {v: k for k, v in map_label.items()}
                model_result = inv_map_label[pred_index]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                model_result = "Unknown"
        else:
            model_result = "Unknown"

        # Dictionary-based prediction
        tokens = preprocessor.preprocessing(text).lower().split()
        dict_result, scores = predict_label_from_tokens(tokens, category_words)
        temp_ban = essential(BAN_WORDS_PATH, tokens)

        # Semantic label
        semantic_label, similarity_score = classify_text_semantically(text)

        # Combine results
        if model_result == dict_result:
            final_label = model_result
        elif (model_result in ["Political", "Gambling", "Adult Content"]) and (dict_result not in ["Political", "Gambling", "Adult Content"]):
            final_label = model_result if temp_ban else dict_result
        elif (model_result not in ["Political", "Gambling", "Adult Content"]) and (dict_result in ["Political", "Gambling", "Adult Content"]):
            final_label = dict_result if temp_ban else model_result
        else:
            final_label = dict_result

        status = not any(word in tokens for word in temp_ban)

        # Display results
        st.subheader("Results")
        st.write("Tokens:", tokens)
        st.write("Hashtags analysis:", hashtag_results if hashtags else "No hashtags found")
        st.write("Model predicted label:", model_result)
        st.write("Dictionary predicted label:", dict_result)
        st.write("Final label:", final_label)
        st.write("Banned words found:", temp_ban)
        st.write("Status:", status)