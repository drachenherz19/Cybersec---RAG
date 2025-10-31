from transformers import pipeline
from keybert import KeyBERT


def generate_keywords(input_text):
    # Load the KeyBERT model using a pre-trained transformer (e.g., 'distilbert-base-nli-mean-tokens')
    model = KeyBERT(model='distilbert-base-nli-mean-tokens')

    # Extract keywords using KeyBERT
    keywords = model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

    # Print the extracted keywords
    print("Relevant Keywords:")
    for keyword, score in keywords:
        print(f"- {keyword} (score: {score:.2f})")


if __name__ == "__main__":
    # Example user input
    user_input = input("Enter a cyber threat or issue description:\n")
    generate_keywords(user_input)
