import streamlit as st
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import evaluate
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "./model"
MODEL_NAME = "google/flan-t5-small"

# -----------------------------
# LOAD TOKENIZER + MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# -----------------------------
# TRAIN MODEL (ONLY ONCE)
# -----------------------------
def train_model():
    st.write("Training model... (this runs once)")

    dataset = load_dataset("banking77")

    def format_data(example):
        return {
            "input": example["text"],
            "output": str(example["label"])
        }

    dataset = dataset.map(format_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def tokenize(example):
        inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)
        targets = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)

        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"]
    )

    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    st.success("Training completed and model saved ✅")

# -----------------------------
# GENERATE RESPONSE
# -----------------------------
def get_response(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True)

    outputs = model.generate(**inputs, max_length=50)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(tokenizer, model):
    st.write("Evaluating model...")

    dataset = load_dataset("banking77", split="test[:50]")

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    for sample in dataset:
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=50)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append([str(sample["label"])])

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references])

    st.write("BLEU Score:", bleu_score)
    st.write("ROUGE Score:", rouge_score)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Customer Support Chatbot + Evaluation 💬")

# Train button
if st.button("Train Model (Run Once)"):
    train_model()

# Load model
tokenizer, model = load_model()

# Chat UI
st.subheader("Chat with Bot")
user_input = st.text_input("Ask your question:")

if user_input:
    response = get_response(user_input, tokenizer, model)
    st.write("Bot:", response)

# Evaluation button
if st.button("Evaluate Model"):
    evaluate_model(tokenizer, model)
