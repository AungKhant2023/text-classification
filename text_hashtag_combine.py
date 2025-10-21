# -------------------- CONFIGURE ENVIRONMENT --------------------
import os
import sys
import logging

# -------------------- REDIRECT LOW-LEVEL LOGS --------------------
# Temporarily suppress stdout/stderr during TF import
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

# Force TensorFlow to CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, ERROR logs

# -------------------- IMPORTS --------------------
import tensorflow as tf
from absl import logging as absl_logging

# Re-enable stdout/stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Suppress absl logs and TensorFlow internal logs
absl_logging.set_verbosity(absl_logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Other imports
import streamlit as st
import pickle
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessor import MyanmarTextPreprocessor
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import unicodedata
from sentence_transformers import SentenceTransformer, util
import torch

# -------------------- UTILITY FUNCTIONS --------------------
def load_category_words_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    category_words = {}
    for col in df.columns:
        words = df[col].dropna().astype(str).tolist()
        category_words[col.strip()] = set(w.strip() for w in words if w.strip())
    return category_words

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

# -------------------- CUSTOM LAYERS --------------------
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons, dropout_rate=0.5, **kwargs):
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
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# -------------------- LABEL MAPPING --------------------
map_label = {
    'Social': 0, 'Entertainment': 1, 'Product&Service': 2, 'Business': 3, 'Sports': 4,
    'Science&Technology': 5, 'Education': 6, 'Culture&History': 7, 'Health': 8,
    'Environmental': 9, 'Political': 10, 'Gambling': 11, 'Adult Content': 12,
}

# -------------------- LOAD RESOURCES --------------------
maxlen = 100
model = load_model('my_transformer_model.h5', custom_objects={
    'TransformerEncoder': TransformerEncoder,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
})

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

dict_path = 'dict-output-v2-4-9-2025.txt'
sw_path = 'sw.txt'
ban_words_path = "ban-words-4-9-2025.txt"
category_words = load_category_words_from_csv('Distinct-Words-21-8-2025.csv')
preprocessor = MyanmarTextPreprocessor(dict_path, sw_path)

# -------------------- HELPER FUNCTIONS --------------------
def normalize_font_style(text):
    return unicodedata.normalize('NFKC', text)

def safe_pad_sequences(texts, tokenizer, maxlen, vocab_limit):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = [[min(token, vocab_limit - 1) for token in seq] for seq in sequences]
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# -------------------- SENTENCE TRANSFORMER --------------------
st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
st_categories = [
    'Social', 'Entertainment', 'Product', 'Service', 'Business', 'Sports',
    'Science', 'Technology', 'History', 'Education', 'Culture',
    'Health', 'Environmental'
]
st_embeddings = st_model.encode(st_categories, convert_to_tensor=True)

def classify_text_semantically(text):
    text = text.lstrip("#").lower()
    text_emb = st_model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(text_emb, st_embeddings)[0]
    best_idx = torch.argmax(sims).item()
    return st_categories[best_idx], sims[best_idx].item()

# -------------------- FASTAPI APP --------------------
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = normalize_font_style(input.text).lower()

    # Hashtags
    raw_text = text
    hashtags = [word for word in raw_text.split() if word.startswith("#")]
    hashtag_results = []
    if hashtags:
        for tag in hashtags:
            label, score = classify_text_semantically(tag)
            hashtag_results.append({
                "hashtag": tag,
                "semantic_label": label,
                "similarity_score": round(score, 4)
            })

    # Preprocessing + Transformer model
    cleaned = preprocessor.preprocessing(text)
    seq_pad = safe_pad_sequences([cleaned], tokenizer, maxlen, vocab_limit=40701)
    pred_probs = model.predict(seq_pad)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    inv_map_label = {v: k for k, v in map_label.items()}
    model_result = inv_map_label[pred_index]

    tokens = preprocessor.preprocessing(text).lower().split()
    dict_result, scores = predict_label_from_tokens(tokens, category_words)
    temp_ban = essential(ban_words_path, tokens)

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

    return {
        "hashtag_results": hashtag_results if hashtags else "No hashtags found",
        "predicted_label_model": model_result,
        "predicted_label_distinct": dict_result,
        "final_label": final_label,
        "status": status,
        "banned_words_found": temp_ban,
        "tokens": tokens
    }