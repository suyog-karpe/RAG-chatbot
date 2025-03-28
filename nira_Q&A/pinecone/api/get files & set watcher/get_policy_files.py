import os
import logging
import aiofiles
import aiohttp
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
import httpx
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Securely retrieve API key
ACCESS_TOKN = os.getenv("ACCESS_TOKN")

# Node.js API URL
NODE_API_UR = os.getenv("NODE_API_UR")

# Base directory to store downloaded files
BASE_DIR = "downloaded_files"

# Ensure base directory exists
os.makedirs(BASE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def download_and_save_file(file_url: str, file_name: str):
    """
    Downloads a file asynchronously and saves it to the BASE_DIR.
    """
    try:
        file_path = os.path.join(BASE_DIR, file_name)

        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await response.read())
                    logger.info(f"File saved: {file_path}")
                    return f"File saved: {file_path}"
                else:
                    logger.error(f"Failed to download {file_url} - Status: {response.status}")
                    return f"Failed to download {file_url}"
    except Exception as e:
        logger.error(f"Error downloading {file_url}: {str(e)}")
        return f"Error: {str(e)}"


@app.get("/fetch-and-save")
async def fetch_and_save():
    """
    Fetches files from Node.js API and saves them in a single folder.
    """
    try:
        if not ACCESS_TOKN:
            raise HTTPException(status_code=500, detail="API Key is missing. Set ACCESS_TOKN in environment variables.")

        headers = {
            "Authorization": f"Bearer {ACCESS_TOKN}"
        }

        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(NODE_API_UR, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from Node.js API")

        data = response.json()
        if "data" not in data:
            raise HTTPException(status_code=400, detail="Invalid response structure")

        tasks = []
        for item in data["data"]:
            file_url = item["filePath"]
            file_name = os.path.basename(urlparse(file_url).path)  # Extract file name from URL

            # Download and save file asynchronously in BASE_DIR
            tasks.append(download_and_save_file(file_url, file_name))

        # Run all downloads concurrently
        results = await asyncio.gather(*tasks)

        return {"status": "success", "saved_files": results}

    except Exception as e:
        logger.error(f"Error fetching/saving data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching/saving data: {str(e)}")


