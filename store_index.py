from src.helper import load_csv, text_split, download__embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
# Retrieve the Pinecone API key and index name from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX')



# Load data from a CSV file located at 'data/data.csv'
extracted_data = load_csv('data\data.csv')
# Split the loaded data into manageable text chunks
text_chunks = text_split(extracted_data)
# Download or retrieve pre-computed embeddings for the text chunks
embeddings = download__embeddings()

# Ensure the index name is explicitly assigned (this line could actually be omitted)
index_name=index_name

# Initialize a Pinecone Vector Store with the extracted text chunks, specifying the index name and the embeddings
# This will be used for storing and searching text chunks in a vectorized format
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], index_name=index_name,embedding=embeddings)