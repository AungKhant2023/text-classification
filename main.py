# Transformer Encoder with attention mask support
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessor import MyanmarTextPreprocessor
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Load category words from CSV
def load_category_words_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    category_words = {}
    for col in df.columns:
        words = df[col].dropna().astype(str).tolist()
        category_words[col.strip()] = set(w.strip() for w in words if w.strip())
    return category_words

# Prediction logic
def predict_label_from_tokens(tokens, category_words):
    score_dict = {}
    for label, keywords in category_words.items():
        matches = set(tokens) & keywords
        score_dict[label] = len(matches)
    predicted_label = max(score_dict, key=score_dict.get)
    print(predicted_label)
    return predicted_label, score_dict

def essential (path,tokens):
    with open(path,'r',encoding="utf-8") as f:
        temp_ban = set()
        for line in f:
            word = line.strip().lower()
            if word:
                temp_ban.add(word)

        filtered_tokens = [token for token in tokens if token in temp_ban]
        
    return filtered_tokens


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

#--------------------------------------------------------------------------
# Load the model and tokenizer
maxlen = 100
model = load_model('finetune_transformer_ver1.2.keras', custom_objects={
    'TransformerEncoder': TransformerEncoder,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
})
with open('finetune_tokenizer_ver1.2.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

dict_path = 'dict-words.txt'
sw_path = 'sw.txt'
ban_words_path = "essential.txt"
category_words = load_category_words_from_csv('Distinct Words.csv')
preprocessor = MyanmarTextPreprocessor(dict_path, sw_path)

def safe_pad_sequences(texts, tokenizer, maxlen, vocab_limit):
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = [[min(token, vocab_limit - 1) for token in seq] for seq in sequences]
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

class TextInput(BaseModel):
    text: str
    
#--------------------------------------------------------------------------
app = FastAPI()
@app.post("/predict")

def predict(input: TextInput):
    # Model part
    cleaned = preprocessor.preprocessing(input.text)
    seq = tokenizer.texts_to_sequences([cleaned])  # reuse the saved tokenizer
    seq_pad = safe_pad_sequences([cleaned], tokenizer, maxlen, vocab_limit=40701)
    # seq_pad =pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    pred_probs = model.predict(seq_pad)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    inv_map_label = {v: k for k, v in map_label.items()}
    pred_label = inv_map_label[pred_index]
    model_result = pred_label # Predittion from Model 
    
    # Distinct Words part    
    tokens = preprocessor.preprocessing(input.text).split()
    label, scores = predict_label_from_tokens(tokens, category_words)
    temp_ban = essential(ban_words_path,tokens)
    dict_result = label # Prediction from Distinct Words
    
    # Condition for final label
    if model_result == dict_result:
        final_label = model_result
    elif (model_result in ["Political","Gambling","Adult Content"]) and (dict_result not in ["Political","Gambling","Adult Content"]):
        if temp_ban :final_label = model_result
        else: final_label = dict_result
    elif (model_result not in ["Political","Gambling","Adult Content"]) and (dict_result in ["Political","Gambling","Adult Content"]):
        if temp_ban : final_label = dict_result
        else: final_label = model_result    
    else:
        final_label = dict_result

    status = "unsafe" if final_label in ["Political","Gambling","Adult Content"] else "safe"
    return {
        "predicted_label From Disticnt": dict_result,
        "predicted_label From Model": model_result,
        "final_label": final_label,
        "status": status
    }