import time
import re
from datasets import load_dataset
from transformers import pipeline

# -----------------------------
# LOAD MODEL (LOCAL)
# -----------------------------
print("Loading model...")
generator = pipeline("text-generation", model="distilgpt2")  # lightweight

# -----------------------------
# LOAD DATASET (GSM8K)
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("gsm8k", "main", split="test[:20]")  # small subset

# -----------------------------
# PROMPT TYPES
# -----------------------------
def zero_shot_prompt(question):
    return f"Q: {question}\nA:"

def few_shot_prompt(question):
    return """Q: If you have 2 apples and buy 3 more, how many apples do you have?
A: 2 + 3 = 5

Q: John has 10 dollars and spends 4. How much is left?
A: 10 - 4 = 6

Q: """ + question + "\nA:"

def cot_prompt(question):
    return f"Q: {question}\nA: Let's think step by step."

# -----------------------------
# EXTRACT NUMERIC ANSWER
# -----------------------------
def extract_number(text):
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else None

# -----------------------------
# DETECT HALLUCINATION PATTERNS
# -----------------------------
def detect_hallucination(output):
    patterns = []

    if "I think" in output or "maybe" in output:
        patterns.append("uncertainty")

    if len(output.split()) > 100:
        patterns.append("over-generation")

    if not re.search(r'\d', output):
        patterns.append("no numeric answer")

    return patterns

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate(prompt):
    start = time.time()
    result = generator(prompt, max_length=150, num_return_sequences=1)
    end = time.time()

    text = result[0]["generated_text"]
    latency = end - start

    return text, latency

# -----------------------------
# EVALUATION LOOP
# -----------------------------
results = []

for i, sample in enumerate(dataset):
    question = sample["question"]
    correct_answer = extract_number(sample["answer"])

    print(f"\nQuestion {i+1}: {question}")

    for method_name, prompt_func in [
        ("Zero-Shot", zero_shot_prompt),
        ("Few-Shot", few_shot_prompt),
        ("CoT", cot_prompt)
    ]:
        prompt = prompt_func(question)

        output, latency = generate(prompt)
        predicted = extract_number(output)

        correct = (predicted == correct_answer)

        hallucinations = detect_hallucination(output)

        results.append({
            "method": method_name,
            "correct": correct,
            "latency": latency,
            "hallucinations": hallucinations
        })

        print(f"\n[{method_name}]")
        print("Output:", output[:200])
        print("Predicted:", predicted, "| Actual:", correct_answer)
        print("Correct:", correct)
        print("Hallucinations:", hallucinations)

# -----------------------------
# AGGREGATE RESULTS
# -----------------------------
summary = {}

for r in results:
    m = r["method"]
    if m not in summary:
        summary[m] = {"total": 0, "correct": 0, "latency": 0, "hallucinations": 0}

    summary[m]["total"] += 1
    summary[m]["latency"] += r["latency"]

    if r["correct"]:
        summary[m]["correct"] += 1

    if len(r["hallucinations"]) > 0:
        summary[m]["hallucinations"] += 1

# -----------------------------
# PRINT FINAL RESULTS
# -----------------------------
print("\n\n===== FINAL SUMMARY =====")

for method, stats in summary.items():
    accuracy = stats["correct"] / stats["total"]
    avg_latency = stats["latency"] / stats["total"]
    halluc_rate = stats["hallucinations"] / stats["total"]

    print(f"\n{method}")
    print("Accuracy:", round(accuracy, 2))
    print("Avg Latency:", round(avg_latency, 2))
    print("Hallucination Rate:", round(halluc_rate, 2))
