from transformers import AutoModel, AutoTokenizer
from keybert import KeyBERT

# Step 1: Load a Hugging Face Model and Tokenizer
# model_name = "distilbert-base-uncased"
model_name = "EleutherAI/gpt-neo-1.3B"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 2: Initialize KeyBERT with the Hugging Face model
keybert_model = KeyBERT(model=model)


def extract_keywords(input_text):
    # Extract keywords using KeyBERT
    keywords = keybert_model.extract_keywords(
            input_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=10
            )
    return keywords


if __name__ == "__main__":
    # Take user input
    user_input = input("Enter a cyber threat or issue description:\n")
    keywords = extract_keywords(user_input)

    # Print the extracted keywords
    print("\nRelevant Keywords:")
    for keyword, score in keywords:
        print(f"- {keyword} (score: {score:.2f})")
