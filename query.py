import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# A simple tokenizer that splits on non-word characters and lowercases the text.
def tokenize(text):
    return re.findall(r'\w+', text.lower())

# List of example documents.
documents = [
    "This is a document about machine learning.",
    "Another document about apples and fruits.",
    "This text is about cooking recipes.",
    "Here we discuss Python programming."
]

# Build vocabulary from the documents.
vocab = {}
for doc in documents:
    for word in tokenize(doc):
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
embedding_dim = 50  # You can adjust this dimension as needed.

# Define an embedding layer using PyTorch.
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Function to convert a document string into a tensor of word indices.
def doc_to_tensor(doc):
    tokens = tokenize(doc)
    indices = [vocab[word] for word in tokens if word in vocab]
    return torch.tensor(indices, dtype=torch.long)

# Function to compute the document embedding as the average of its word embeddings.
def get_doc_embedding(doc):
    token_tensor = doc_to_tensor(doc)
    if token_tensor.nelement() == 0:
        return torch.zeros(embedding_dim)
    # Get embeddings for all tokens and compute their mean.
    embeddings = embedding_layer(token_tensor)
    doc_embedding = embeddings.mean(dim=0)
    return doc_embedding

# Precompute embeddings for all documents.
doc_embeddings = [get_doc_embedding(doc) for doc in documents]

# Function to find the document most similar to the given query.
def find_best_document(query):
    query_embedding = get_doc_embedding(query)
    best_score = -1.0
    best_doc = None

    for doc, doc_emb in zip(documents, doc_embeddings):
        # Use cosine similarity to compare embeddings.
        if query_embedding.norm() == 0 or doc_emb.norm() == 0:
            similarity = 0
        else:
            similarity = F.cosine_similarity(query_embedding.unsqueeze(0), doc_emb.unsqueeze(0)).item()
        if similarity > best_score:
            best_score = similarity
            best_doc = doc
    return best_doc, best_score

if __name__ == '__main__':
    # Example query string.
    query = "Tell me about learning algorithms."
    best_doc, score = find_best_document(query)
    
    print("Best matching document:")
    print(best_doc)
    print("Cosine similarity score:", score)
