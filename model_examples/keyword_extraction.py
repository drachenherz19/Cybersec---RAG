import os

import requests
import openai
from dotenv import load_dotenv


api_key = "szzzz"

if not api_key:
    raise ValueError("API Key not found. Please set it in the .env file.")

# Initialize the OpenAI client
openai.api_key = api_key


def extract_keywords_with_gpt(user_input):
    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Extract keywords from: '{user_input}'"}],
                temperature=0.7
                )
        keywords = response['choices'][0]['message']['content']
        return keywords
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None



def main():
    user_input = input("Describe your issue: ")
    print("\nExtracting keywords from your input...\n")
    keywords = extract_keywords_with_gpt(user_input)
    if keywords:
        print(f"Extracted Keywords: {keywords}")
    else:
        print("No keywords extracted.")


if __name__ == "__main__":
    main()
