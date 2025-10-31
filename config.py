import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OTX_API_KEY = os.getenv("OTX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DATABASE = os.getenv("DATABASE")
SSLMODE = os.getenv("SSLMODE")