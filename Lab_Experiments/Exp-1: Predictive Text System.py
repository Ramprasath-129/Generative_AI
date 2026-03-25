import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter

# Download necessary datasets
nltk.download('brown')
nltk.download('universal_tagset')

class PredictiveModel:
    def __init__(self):
        # transitions[prev_word] = Counter(next_word)
        self.bigram_counts = defaultdict(Counter)
        # tag_transitions[prev_tag] = Counter(current_tag)
        self.tag_counts = defaultdict(Counter)
        # word_given_tag[tag] = Counter(word)
        self.emission_counts = defaultdict(Counter)

    def train(self):
        print("Training on Brown Corpus...")
        # Use universal tagset for simpler HMM states
        tagged_sentences = brown.tagged_sents(tagset='universal')
        
        for sentence in tagged_sentences:
            for i in range(len(sentence) - 1):
                word_curr, tag_curr = sentence[i]
                word_next, tag_next = sentence[i+1]
                
                # N-Gram logic
                self.bigram_counts[word_curr.lower()][word_next.lower()] += 1
                
                # HMM logic (Transitions and Emissions)
                self.tag_counts[tag_curr][tag_next] += 1
                self.emission_counts[tag_next][word_next.lower()] += 1
        print("Training complete.")

    def predict_next_word(self, current_word, top_n=3):
        current_word = current_word.lower()
        
        # 1. Get candidates from N-Gram
        candidates = self.bigram_counts.get(current_word, {})
        
        if not candidates:
            return "Context not found. Try a common word like 'the' or 'he'."

        # 2. Sort by frequency
        # In a full HMM, we'd use Viterbi, but for next-word prediction,
        # we prioritize words that frequently follow the input.
        sorted_predictions = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_predictions[:top_n]]

# Initialize and run
model = PredictiveModel()
model.train()

# Test it out
context = "the"
predictions = model.predict_next_word(context)
print(f"\nNext word predictions for '{context}': {predictions}")
