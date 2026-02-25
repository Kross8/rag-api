from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Pinecone client ONCE when the server starts
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
_index = None

def get_pinecone_index():
    global _index
    if _index is None:
        # Connect to the index only if we haven't already
        _index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    return _index