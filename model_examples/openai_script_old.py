import requests
from textblob import TextBlob
import openai
from config import OPENAI_API_KEY

'''# Check if the API key is being loaded correctly
if not OPENAI_API_KEY:
    print("API Key not found!")
else:
    print("API Key loaded successfully.")'''


# Spell-checking function
def check_spelling(user_query):
    """
    Custom spell checker that preserves technical and security-related terms
    """
    # List of terms to preserve (never correct)
    preserve_terms = {
            'website', 'web', 'server', 'ddos', 'dos', 'sql', 'xss', 'rce',
            'apache', 'nginx', 'iis', 'ftp', 'ssh', 'http', 'https',
            'api', 'database', 'mysql', 'postgresql', 'php', 'java',
            'linux', 'windows', 'unix', 'azure', 'aws', 'ransomware'
            }

    # Split query into words
    words = user_query.split()

    # Check if any word should be preserved
    for word in words:
        if word.lower() in preserve_terms:
            return None  # Don't suggest any correction if we find technical terms

    # Only if no technical terms found, try basic spell check
    blob = TextBlob(user_query)
    corrected_query = str(blob.correct())

    # Return correction only if significantly different
    if corrected_query.lower() != user_query.lower():
        return corrected_query
    return None


def generate_keywords(user_query):
    """
    Extract cybersecurity keywords from user query that are relevant to CVE/NVD database
    """
    # Define common CVE-relevant technical terms
    cve_terms = {
            # Attack Types
            'sql injection': 'sql-injection',
            'xss': 'xss',
            'csrf': 'csrf',
            'buffer overflow': 'buffer-overflow',
            'ddos': 'ddos',
            'dos': 'dos',
            'rce': 'rce',
            'remote code execution': 'rce',
            'privilege escalation': 'privilege-escalation',
            'command injection': 'command-injection',
            'directory traversal': 'directory-traversal',
            'file inclusion': 'file-inclusion',

            # Infrastructure/Software
            'apache': 'apache',
            'nginx': 'nginx',
            'iis': 'iis',
            'windows': 'windows',
            'linux': 'linux',
            'web server': 'web-server',
            'database': 'database',
            'mysql': 'mysql',
            'postgresql': 'postgresql',
            'active directory': 'active-directory',
            'php': 'php',
            'java': 'java',
            'openssl': 'openssl',
            'ssh': 'ssh',
            'ftp': 'ftp',

            # Common Vulnerability Terms
            'vulnerability': 'vulnerability',
            'exploit': 'exploit',
            'attack': 'attack',
            'security': 'security',
            'authentication': 'authentication',
            'authorization': 'authorization',
            'bypass': 'bypass',
            'overflow': 'overflow',
            'injection': 'injection',
            'credentials': 'credentials',
            'password': 'password',

            # Malware Types
            'ransomware': 'ransomware',
            'malware': 'malware',
            'virus': 'virus',
            'trojan': 'trojan',
            'backdoor': 'backdoor'
            }

    prompt = f"""
    Analyze this query and situation what the user is facing and give me ONLY the cybersecurity-relevant keywords that might appear in CVE/NVD database entries:
    Query: '{user_query}'
    
    Guidelines:
    - Understand the user problem in Cybersecurity World and give relevant keywords
    - Focus on technical terms that would appear in CVE entries
    - Include affected software/systems if mentioned
    - Include attack types if specified
    - Return as comma-separated list
    - Modify the keywords as you know would appear in CVE entries
    
    Example good output for "Our Apache web server is under SQL injection attack":
    apache, web-server, sql-injection, attack
    """

    print('Trying prompt : ', prompt)

    try:
        response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                        {
                                "role": "system",
                                "content": """You are a cybersecurity keyword extractor specialized in CVE/NVD database terminology. 
                    Only extract terms that would be relevant for searching CVE entries.
                    Preserve technical terms exactly as they appear."""
                                },
                        {"role": "user", "content": prompt}
                        ],
                max_tokens=50,
                temperature=0.1  # Low temperature for consistent output
                )

        # Get keywords from response
        raw_keywords = response.choices[0].message.content.strip().split(',')

        # Clean and standardize keywords
        keywords = []
        for keyword in raw_keywords:
            keyword = keyword.strip().lower()
            # If keyword is in our CVE terms, use the standardized version
            if keyword in cve_terms:
                keywords.append(cve_terms[keyword])
            else:
                # Check if any multi-word term contains this keyword
                found = False
                for term in cve_terms:
                    if keyword in term:
                        keywords.append(cve_terms[term])
                        found = True
                        break
                if not found:
                    keywords.append(keyword)

        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)

        # Join keywords with plus signs for API compatibility
        return "+".join(unique_keywords)

    except Exception as e:
        print(f"Error generating keywords: {e}")
        # Return basic keywords extracted from query using cve_terms
        fallback_keywords = []
        query_lower = user_query.lower()
        for term in cve_terms:
            if term in query_lower:
                fallback_keywords.append(cve_terms[term])
        return "+".join(fallback_keywords) if fallback_keywords else "security+issue"


def fetch_data_from_api(user_query):
    # Step 1: Generate keywords
    formatted_query = generate_keywords(user_query)

    # Step 2: Construct the API URL with the formatted query
    base_url = "https://api.openai.com/v1/chat/completions"
    api_url = f"{base_url}?query={formatted_query}"

    # Step 3: Set up the headers for authentication
    headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
            }

    # Step 4: Set up the body content
    payload = {
            "model": "gpt-3.5-turbo",  # Use the model you want to interact with
            "messages": [
                    {"role": "user", "content": user_query}
                    ]
            }

    print('data query : ', formatted_query)
    # Step 3: Send a GET request to the API
    response = requests.post(api_url, headers=headers, json=payload)

    # Step 4: Handle and print the API response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']  # Assuming the API returns JSON data
    else:
        return f"Error: {response.status_code}"


