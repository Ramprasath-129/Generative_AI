import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# LOAD PDF
# -----------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


# -----------------------------
# SPLIT TEXT INTO CHUNKS
# -----------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
def create_embeddings(chunks, model):
    return model.encode(chunks)


# -----------------------------
# BUILD FAISS INDEX
# -----------------------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index


# -----------------------------
# RETRIEVE RELEVANT CHUNKS
# -----------------------------
def retrieve(query, model, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)

    retrieved = [chunks[i] for i in indices[0]]
    return retrieved


# -----------------------------
# GENERATE ANSWER (LOCAL LLM)
# -----------------------------
def generate_answer(context, question, generator):
    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt, max_length=300, do_sample=False)
    return result[0]["generated_text"]


# -----------------------------
# MAIN RAG PIPELINE
# -----------------------------
def main():
    # Load models
    print("Loading models...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text-generation", model="distilgpt2")

    # Load PDF
    print("Loading PDF...")
    text = load_pdf("/content/file.pdf")

    # Chunk text
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    # Create embeddings
    embeddings = create_embeddings(chunks, embed_model)

    # Build FAISS index
    index = build_index(embeddings)

    print("RAG system ready ✅")

    # Query loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        retrieved_chunks = retrieve(query, embed_model, index, chunks)

        context = "\n".join(retrieved_chunks)

        # Generate answer
        answer = generate_answer(context, query, generator)

        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
