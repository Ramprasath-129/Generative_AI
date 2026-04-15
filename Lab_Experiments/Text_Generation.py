import time
from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download tokenizer
nltk.download('punkt')

# -----------------------------
# PROMPT
# -----------------------------
PROMPT = "Explain the concept of machine learning in simple terms."

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")

gpt2 = pipeline("text-generation", model="gpt2")
distilgpt2 = pipeline("text-generation", model="distilgpt2")

# -----------------------------
# GENERATION FUNCTION
# -----------------------------
def generate(model, name, prompt):
    start = time.time()
    output = model(prompt, max_length=150, num_return_sequences=1)
    end = time.time()

    text = output[0]['generated_text']
    latency = end - start

    return {
        "name": name,
        "text": text,
        "latency": latency,
        "length": len(text.split())
    }

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(reference, generated):
    ref_tokens = reference.split()
    gen_tokens = generated.split()

    bleu = sentence_bleu([ref_tokens], gen_tokens)

    return bleu

# -----------------------------
# RUN COMPARISON
# -----------------------------
print("\nPROMPT:\n", PROMPT)
print("=" * 60)

results = []

# Generate outputs
results.append(generate(gpt2, "GPT-2", PROMPT))
results.append(generate(distilgpt2, "DistilGPT-2", PROMPT))

# Use GPT-2 as reference
reference = results[0]["text"]

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
for res in results:
    print(f"\n--- {res['name']} Output ---")
    print(res["text"])
    print("\nMetrics:")
    print("Latency:", res["latency"])
    print("Length:", res["length"])

    bleu = evaluate(reference, res["text"])
    print("BLEU Score:", bleu)

print("\nComparison Completed ✅")
