from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load environment variables
load_dotenv()

Pinecone_api = os.getenv("Pinecone_api")
Pinecone_index_name = os.getenv("Pinecone_index_name")

# Ensure the embedding model is downloaded locally
snapshot_download(repo_id="intfloat/e5-large-v2")

# Initialize Pinecone
pc = Pinecone(api_key=Pinecone_api, environment="us-west1-gcp")
index = pc.Index(Pinecone_index_name)

# Load the embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Ollama model name
MODEL_NAME = "qwen2.5:0.5b"  # Ensure this model is available locally in Ollama

# FastAPI app
app = FastAPI()

# Store the last response
chatbot_last_response = {}

# Request model for user queries
class QueryRequest(BaseModel):
    query: str

# Function to retrieve relevant documents from Pinecone
def retrieve_relevant_docs(query, top_k=5):
    """
    Fetches the top_k most relevant chunks from Pinecone.
    """
    query_vector = model.encode(query).tolist()
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Extract retrieved text
    retrieved_texts = [match["metadata"]["text"] for match in response["matches"]]
    return retrieved_texts

# Function to generate response using Ollama
def generate_response(query):
    """Retrieves relevant chunks from Pinecone and generates a response using Qwen2.5 3B."""
    retrieved_texts = retrieve_relevant_docs(query)

    # Construct the prompt using retrieved context
    context = "\n".join(retrieved_texts)
    
    prompt = f"""
    **Role**: You are an HR policy compliance assistant. Respond ONLY using verbatim text from this context:
    {context}

    # **Core Rules**:
    1. **Truthfulness**  
    If the answer isn't 100% contained in the context, respond:  
    "Sorry, I don’t have that information."

    2. **No External Knowledge**  
    Never reference people/movies/concepts outside the provided text.

    3. **Literal Matching**  
    - Match query terms EXACTLY to context (case-insensitive)  
    - Ignore partial matches ("Yog" ≠ "Yoga")  

    4. **Response Format**  
    - Max 2 sentences  
    - No markdown/bullets unless context uses them  
    - Never add explanations

    # **Validation Process**  
    Before responding, ASK:  
    1. Does the context contain the EXACT information needed?  
    2. Is this about company policies/employees?  
    3. Would answering require any assumptions?  

    # **Final Output**  
    If all 3 checks pass → Quote relevant context fragment  
    Else → "Sorry, I don’t have that information."
    """



    # Call Ollama's Qwen2.5 3B model with a strict system instruction
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI that strictly follows company policies. Do not add any extra information beyond what is retrieved from the policy document."},
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']

# API endpoint to send query to chatbot
@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Receives a user query and stores the chatbot's response."""
    try:
        query = request.query
        response = generate_response(query)

        # Store the response
        chatbot_last_response["query"] = query
        chatbot_last_response["response"] = response

        return {"message": "Query received. Response is being processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to retrieve chatbot response
@app.get("/response")
async def get_chatbot_response():
    """Returns the last response from the chatbot."""
    if "response" not in chatbot_last_response:
        raise HTTPException(status_code=404, detail="No response available yet. Please send a query first.")

    return chatbot_last_response
