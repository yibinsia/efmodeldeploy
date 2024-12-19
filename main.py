from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import json
import logging
import pickle
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path

# Setup Environment and Logging
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "embedding_log.log", mode='w')
    ]
)

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Load the pre-trained model
MODEL_PATH = BASE_DIR / "NLP/model/XGB-ethnicityFinder_18000.pkl"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# FastAPI App
app = FastAPI()


# Request Body Schema
class NameItem(BaseModel):
    name: str


@app.post("/api/ethnicityFinder/")
async def batch_classification(names: List[NameItem]):
    """
    Endpoint to classify a batch of names. 
    Input: List of dictionaries with a 'name' key.
    """
    try:
        # Extract names from input
        name_list = [item.name for item in names]

        if not name_list:
            raise HTTPException(status_code=400, detail="The input list cannot be empty.")

        # Generate embeddings
        def get_embedding_batch(texts: List[str]) -> List[List[float]]:
            """Generate embeddings for a batch of names."""
            texts = [text.replace("\n", " ") for text in texts]
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    # Process embedding for each row
                    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    logging.info(f"Successfully generated embedding for row {i + 1}: '{text[:50]}...'")
                except Exception as e:
                    logging.error(f"Error generating embedding for row {i + 1}: '{text[:50]}...' | Error: {e}")
                    embeddings.append([0.0] * 1536)  # Append empty embedding for failed rows
            return embeddings

        logging.info(f"Generating embeddings for {len(name_list)} names...")
        embeddings = get_embedding_batch(name_list)

        # Predict ethnicity
        predictions = model.predict(embeddings)

        # Prepare response
        results = [{"name": name, "predicted_ethnicity": pred} for name, pred in zip(name_list, predictions)]

        # Save results to a JSON file
        output_path = BASE_DIR / "results/batch_predictions.json"
        os.makedirs(BASE_DIR / "results", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        logging.info(f"Batch predictions saved to {output_path}")

        return {"message": "Batch classification successful", "results": results}

    except Exception as e:
        logging.error(f"Error during batch classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "OK", "message": "Ethnicity Finder API is running."}
