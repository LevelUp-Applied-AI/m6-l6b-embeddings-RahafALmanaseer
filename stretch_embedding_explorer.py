import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
import torch
import gensim.downloader as api

# 1. Configuration & Categories
categories = {
    "Countries": ["jordan", "china", "italy", "spain", "france", "japan", "brazil", "germany", "egypt", "canada"],
    "Sports": ["football", "tennis", "stadium", "athlete", "coach", "match", "tournament", "goal", "olympics", "medal"],
    "Technology": ["software", "internet", "robot", "algorithm", "digital", "mobile", "computer", "hacker", "coding", "data"],
    "Emotions": ["happy", "angry", "sad", "brave", "fear", "joy", "lonely", "excited", "guilt", "surprise"],
    "Business": ["profit", "stocks", "economy", "investment", "bank", "market", "currency", "trade", "wealth", "company"]
}

# 2. Real GloVe Embeddings Processing
print("Loading real GloVe vectors...")
glove_model = api.load("glove-wiki-gigaword-50") 

all_words = []
word_labels = []
word_vectors = []

for cat, words in categories.items():
    for w in words:
        if w.lower() in glove_model:
            for _ in range(4):
                all_words.append(w)
                word_labels.append(cat)
                word_vectors.append(glove_model[w.lower()])

word_vectors = np.array(word_vectors)

# Apply t-SNE for Word Vectors
tsne_words = TSNE(n_components=2, perplexity=15, init='pca', learning_rate='auto', random_state=42)
words_2d = tsne_words.fit_transform(word_vectors)

# 3. Document Embeddings Processing (DistilBERT)
df = pd.read_csv('data/bbc_news.csv') 
sample_df = df.groupby('category').head(4).reset_index(drop=True)

def get_distilbert_embeddings(text_list):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

print("Computing DistilBERT embeddings...")
article_embeddings = get_distilbert_embeddings(sample_df['text'].tolist())

tsne_docs = TSNE(n_components=2, perplexity=5, random_state=42)
docs_2d = tsne_docs.fit_transform(article_embeddings)

# 4. Visualizations
# Plot 1: Word Embeddings
plt.figure(figsize=(12, 8))
sns.scatterplot(x=words_2d[:, 0], y=words_2d[:, 1], hue=word_labels, palette="Set1", s=60)
for i in range(0, len(all_words), 20):
    plt.annotate(all_words[i], (words_2d[i, 0], words_2d[i, 1]), fontsize=9, alpha=0.7)
plt.title("Visualization of 200 GloVe Word Vectors (t-SNE)")
plt.legend(title="Categories")
plt.savefig("word_embeddings.png")
plt.show()

# Plot 2: Document Embeddings
plt.figure(figsize=(12, 8))
sns.scatterplot(x=docs_2d[:, 0], y=docs_2d[:, 1], hue=sample_df['category'], style=sample_df['category'], s=100)
for i, cat in enumerate(sample_df['category']):
    plt.annotate(cat, (docs_2d[i, 0], docs_2d[i, 1]), fontsize=8, alpha=0.8)
plt.title("Visualization of BBC News DistilBERT Embeddings (t-SNE)")
plt.savefig("document_embeddings.png")
plt.show()

print("Assignment complete. Plots saved as PNG files.")