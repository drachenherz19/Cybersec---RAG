import openai as ai
from config import OPENAI_API_KEY

ai.api_key = OPENAI_API_KEY

if not ai.api_key:
    raise ValueError("API Key not found. Please set it in the .env file.")


#
#
# def generate_gpt3_response(user_text, print_output=False):
#     """
#     Query OpenAI GPT-3 for the specific key and get back a response
#     :type user_text: str the user's text to query for
#     :type print_output: boolean whether or not to print the raw output JSON
#     """
#     completions = ai.Completion.create(
#             engine='text-davinci-003',  # Determines the quality, speed, and cost.
#             temperature=0.5,  # Level of creativity in the response
#             prompt=user_text,  # What the user typed in
#             max_tokens=100,  # Maximum tokens in the prompt AND response
#             n=1,  # The number of completions to generate
#             stop=None,  # An optional setting to control response generation
#             )
#
#     # Displaying the output can be helpful if things go wrong
#     if print_output:
#         print(completions)
#
#     # Return the first choice's text
#     return completions.choices[0].text


def extract_keywords_with_gpt(user_input):
    try:
        response = ai.ChatCompletion.create(
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
