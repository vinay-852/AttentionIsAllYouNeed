from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def inputEmbedding(input_text):
    return model.encode(input_text)