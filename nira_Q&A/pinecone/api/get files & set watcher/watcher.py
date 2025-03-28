import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

Pinecone_api = os.getenv("Pinecone_api")
Pinecone_index_name = os.getenv("Pinecone_index_name")

# Initialize Pinecone Client
pc = Pinecone(api_key=Pinecone_api)
index = pc.Index(Pinecone_index_name)

# 🔹 Download necessary NLTK models
nltk.download('punkt')

# Folder to monitor for new files
WATCH_FOLDER = os.path.abspath("downloaded_files")

# 🔹 Load SentenceTransformer Model
embedding_model = SentenceTransformer("intfloat/e5-large-v2")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_text_file(file_path):
    """Reads text content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text_by_sentence(text, max_chunk_size=500):
    """Splits text into chunks based on sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_embeddings(text_chunks):
    """Generates embeddings for text chunks using intfloat/e5-large-v2."""
    instruction = "query: "  # Required prefix for e5 embeddings
    embeddings = embedding_model.encode([instruction + chunk for chunk in text_chunks])
    return embeddings


def store_in_pinecone(file_name, chunks, embeddings):
    """Stores chunks and their embeddings in Pinecone."""
    vector_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_data.append({
            "id": f"{file_name}_{i}",
            "values": embedding.tolist(),
            "metadata": {"source": file_name, "text": chunk}
        })

    index.upsert(vectors=vector_data)
    logger.info(f"Stored {len(vector_data)} vectors for {file_name} in Pinecone")


def process_new_file(file_path):
    """Extracts text, generates embeddings, and stores them in Pinecone."""
    if not file_path.endswith(".txt"):
        logger.info(f"Skipping {file_path}: Not a text file.")
        return

    text = read_text_file(file_path)
    if not text:
        logger.info(f"Skipping {file_path}: No extractable text.")
        return

    chunks = chunk_text_by_sentence(text)
    embeddings = generate_embeddings(chunks)
    store_in_pinecone(os.path.basename(file_path), chunks, embeddings)


class FileHandler(FileSystemEventHandler):
    """Watchdog Event Handler to detect new files."""
    
    def on_created(self, event):
        if event.is_directory:
            return
        logger.info(f"New file detected: {event.src_path}")
        process_new_file(event.src_path)


def watch_folder():
    """Starts monitoring the folder for new files."""
    observer = Observer()
    event_handler = FileHandler()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    logger.info(f"Watching folder: {WATCH_FOLDER}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    logger.info("Starting File Watcher...")
    watch_folder()

