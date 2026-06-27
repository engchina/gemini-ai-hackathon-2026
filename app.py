import os
import logging
import asyncio
import json
import re
import urllib.parse
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search, FunctionTool

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format='%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Detailed HTTP Request & Response Logging Middleware
import time

@app.before_request
def log_request_info():
    request.start_time = time.time()
    body = ""
    if request.is_json:
        try:
            body = json.dumps(request.get_json())
        except Exception:
            body = "<invalid json>"
    elif request.form:
        body = str(dict(request.form))
    elif request.files:
        body = f"Files: {list(request.files.keys())}"
    
    logger.info(f"==> HTTP Request: {request.method} {request.path} | Remote IP: {request.remote_addr} | Payload: {body}")

@app.after_request
def log_response_info(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        duration_ms = f"{duration * 1000:.2f}ms"
    else:
        duration_ms = "unknown"
        
    status = response.status
    response_body = ""
    if response.is_json:
        try:
            response_body = response.get_data(as_text=True)
            if len(response_body) > 1000:
                response_body = response_body[:1000] + "... (truncated)"
        except Exception:
            response_body = "<failed to read json>"
            
    logger.info(f"<== HTTP Response: {request.method} {request.path} | Status: {status} | Duration: {duration_ms} | Response Body: {response_body}")
    return response

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Initialize the GenAI Client
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Define the custom python function for search
def search_google_for_info(query: str) -> str:
    """
    Search Google for information about a product, recipe, or general details.
    
    Args:
        query: The search query to look up on Google.
        
    Returns:
        A text summary of search results from Google.
    """
    logger.info(f"Custom tool search_google_for_info executing query: '{query}'")
    if not genai_client:
        logger.error("Gemini client not initialized. Cannot perform custom tool search.")
        return "Gemini client not initialized. Cannot search."
    
    import time
    prompt = f"Search Google for: '{query}'. Provide a concise summary of the findings."
    for attempt in range(4):
        try:
            logger.info(f"Invoking Custom Search tool via Gemini (attempt {attempt+1}/4) with prompt: '{prompt}'")
            start_time = time.time()
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            elapsed = time.time() - start_time
            logger.info(f"Custom Search tool response received in {elapsed:.2f}s (attempt {attempt+1}/4)")
            result_text = response.text or "No results found on Google."
            logger.debug(f"Custom Search tool response text: '{result_text}'")
            return result_text
        except Exception as e:
            if attempt < 3:
                wait_time = 1.5 * (2 ** attempt)
                logger.warning(f"Error in search_google_for_info (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.exception("Failed search_google_for_info after multiple attempts:")
                return f"Error searching Google: {e}"
    return "Failed to retrieve search results after multiple attempts due to temporary service unavailability."

# Wrap python function as FunctionTool
search_tool = FunctionTool(func=search_google_for_info)

# Initialize Google ADK Agents
# 1. Multimodal Item Extractor Agent
extractor_agent = LlmAgent(
    name="costco_item_extractor",
    model=GEMINI_MODEL,
    instruction="""You are a Costco Product Recognition Expert.
You will be provided with an image which can be a shopping receipt, or a photo of one or more products.
Analyze the image and recognize all the purchased or shown items.
For each item, identify:
1. A unique product ID (like the Costco item number/SKU/barcode if visible. If not visible, generate a unique 6-digit number starting with 9).
2. The product name in Japanese.
Output the result ONLY as a valid JSON array of objects, where each object has "id" and "name" keys.
Do not include any markdown backticks, formatting, or conversational text.
Example output:
[{"id": "583495", "name": "ココナッツポークカレー"}, {"id": "902144", "name": "トイレットペーパー"}]"""
)
extractor_runner = InMemoryRunner(agent=extractor_agent)
extractor_runner.auto_create_session = True

# 2. AI Shopping Assistant Agent
assistant_agent = LlmAgent(
    name="costco_assistant",
    model=GEMINI_MODEL,
    instruction="""You are a helpful Costco Shopping Assistant. 
You help users with the items they scanned (receipts or product packagings).
You can suggest recipes, storage methods, nutritional facts, or price comparisons.
You have access to Google Search via the search_google_for_info tool to find accurate info about products or recipes.
When searching Google for details about any scanned product, you must use the query format 'コストコ <Product ID>' (using the 5-6 digit product ID/code instead of its name).
Always answer in Japanese. Be friendly, polite, and professional.""",
    tools=[search_tool]
)
assistant_runner = InMemoryRunner(agent=assistant_agent)
assistant_runner.auto_create_session = True

def extract_json_array(text):
    text = text.strip()
    # Try to find a JSON array block
    match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return []

def run_extractor_agent(image_bytes, mime_type):
    logger.info(f"Starting run_extractor_agent. Mime type: {mime_type}, Image size: {len(image_bytes)} bytes")
    async def _run():
        msg = types.Content(parts=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text="Please recognize and list the purchased/shown Costco items in this image in JSON format.")
        ])
        last_exception = None
        for attempt in range(4):
            try:
                text_parts = []
                logger.info(f"Invoking extractor_agent runner (attempt {attempt+1}/4)")
                start_time = time.time()
                async for event in extractor_runner.run_async(
                    user_id="default_user",
                    session_id="extractor_session",
                    new_message=msg
                ):
                    logger.debug(f"Extractor Agent Event: {event}")
                    if event.error_message:
                        raise Exception(event.error_message)
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                text_parts.append(part.text)
                elapsed = time.time() - start_time
                result_text = "".join(text_parts)
                logger.info(f"Extractor Agent succeeded in {elapsed:.2f}s (attempt {attempt+1}/4). Result: '{result_text}'")
                return result_text
            except Exception as e:
                last_exception = e
                if attempt < 3:
                    wait_time = 1.5 * (2 ** attempt)
                    logger.warning(f"Error in run_extractor_agent (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Extractor Agent failed after {attempt+1} attempts: {e}")
                    raise e
        raise last_exception
    return asyncio.run(_run())

def run_chat_agent(message, session_id, scanned_items):
    logger.info(f"Starting run_chat_agent. Session ID: {session_id}, Scanned items count: {len(scanned_items)}")
    async def _run():
        items_str = ", ".join([f"{item.get('name')} (ID: {item.get('id')})" for item in scanned_items])
        full_prompt = f"Scanned items currently in inventory: [{items_str}]. User asks: {message}"
        logger.info(f"Chat Agent Prompt: '{full_prompt}'")
        
        msg = types.Content(parts=[types.Part.from_text(text=full_prompt)])
        last_exception = None
        for attempt in range(4):
            try:
                text_parts = []
                logger.info(f"Invoking assistant_agent runner (attempt {attempt+1}/4)")
                start_time = time.time()
                async for event in assistant_runner.run_async(
                    user_id="default_user",
                    session_id=session_id,
                    new_message=msg
                ):
                    logger.debug(f"Chat Agent Event: {event}")
                    if event.error_message:
                        raise Exception(event.error_message)
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                text_parts.append(part.text)
                elapsed = time.time() - start_time
                result_text = "".join(text_parts)
                logger.info(f"Chat Agent succeeded in {elapsed:.2f}s (attempt {attempt+1}/4). Result: '{result_text}'")
                return result_text
            except Exception as e:
                last_exception = e
                if attempt < 3:
                    wait_time = 1.5 * (2 ** attempt)
                    logger.warning(f"Error in run_chat_agent (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Chat Agent failed after {attempt+1} attempts: {e}")
                    raise e
        raise last_exception
    return asyncio.run(_run())

def google_custom_image_search(query, api_key, cx):
    logger.info(f"Starting google_custom_image_search query: '{query}'")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "searchType": "image",
        "num": 6
    }
    import time
    for attempt in range(4):
        try:
            logger.info(f"Requesting Google Custom Search API (attempt {attempt+1}/4) for query: '{query}'")
            start_time = time.time()
            response = requests.get(url, params=params, timeout=10)
            elapsed = time.time() - start_time
            logger.debug(f"Google Custom Search API response status: {response.status_code} in {elapsed:.2f}s")
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            logger.debug(f"Google Custom Search API returned {len(items)} items")
            
            results = []
            for item in items:
                image_info = item.get("image", {})
                results.append({
                    "title": item.get("title", ""),
                    "domain": item.get("displayLink", ""),
                    "image_url": item.get("link", ""),
                    "thumbnail_url": image_info.get("thumbnailLink") or item.get("link", ""),
                    "referer_url": image_info.get("contextLink", "")
                })
            logger.info(f"Google Custom Search API parsed results: {results}")
            return results
        except Exception as e:
            err_msg = str(e)
            if attempt < 3:
                wait_time = 1.5 * (2 ** attempt)
                logger.warning(f"Error in Google Custom Search (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed Google Custom Search after multiple attempts: {e}")
                return []

def google_grounding_image_search(query, api_key):
    logger.info(f"Starting google_grounding_image_search query: '{query}'")
    if not genai_client:
        logger.error("Gemini client not initialized. Cannot perform grounding image search.")
        return []
        
    prompt = f"Search Google for '{query}' and tell me the name of the product and its details."
    
    import time
    response = None
    for attempt in range(4):
        try:
            logger.info(f"Invoking Gemini Grounding Search (attempt {attempt+1}/4) for query: '{query}'")
            start_time = time.time()
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            elapsed = time.time() - start_time
            logger.info(f"Gemini Grounding Search response received in {elapsed:.2f}s (attempt {attempt+1}/4)")
            break
        except Exception as e:
            if attempt < 3:
                wait_time = 1.5 * (2 ** attempt)
                logger.warning(f"Error in google_grounding_image_search (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed google_grounding_image_search after multiple attempts: {e}")
                return []
                
    if not response:
        logger.error("Failed to complete grounding image search after multiple retries.")
        return []
        
    try:
        meta = response.candidates[0].grounding_metadata
        if not meta or not meta.grounding_chunks:
            logger.info("No grounding metadata or chunks found in Gemini response.")
            return []
            
        logger.debug(f"Grounding chunks: {meta.grounding_chunks}")
        results = []
        for chunk in meta.grounding_chunks[:4]: # limit to top 4 results
            if not chunk.web:
                continue
            title = chunk.web.title
            uri = chunk.web.uri
            
            # Fetch the web page to extract image (og:image)
            image_url = None
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}
                logger.debug(f"Grounding search fetching webpage: '{uri}' to extract image")
                page_res = requests.get(uri, headers=headers, timeout=5)
                logger.debug(f"Fetch response status: {page_res.status_code} for webpage: '{uri}'")
                if page_res.status_code == 200:
                    # Look for og:image
                    og_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', page_res.text)
                    if not og_match:
                        og_match = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', page_res.text)
                    
                    if og_match:
                        image_url = og_match.group(1)
                        logger.debug(f"Found og:image: '{image_url}' from '{uri}'")
                    else:
                        # Regex for any valid image URL inside img tag src
                        img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|png|jpeg))["\']', page_res.text)
                        for img in img_matches:
                            if not any(x in img.lower() for x in ['logo', 'icon', 'avatar', 'sprite']):
                                image_url = urllib.parse.urljoin(uri, img)
                                logger.debug(f"Found fallback image: '{image_url}' from img tag in '{uri}'")
                                break
            except Exception as e:
                logger.debug(f"Failed to fetch or parse image from '{uri}': {e}")
                pass
                
            results.append({
                "title": title,
                "domain": urllib.parse.urlparse(uri).netloc,
                "image_url": image_url or "/static/placeholder.svg",
                "thumbnail_url": image_url or "/static/placeholder.svg",
                "referer_url": uri
            })
        logger.info(f"google_grounding_image_search parsed results: {results}")
        return results
    except Exception as e:
        logger.error(f"Error in grounding image search: {e}")
        return []

def select_most_correct_image(results):
    """
    Selects the single most correct and relevant image from search results.
    Avoids placeholders, logos, and icons, prioritizing the highest-ranked search result.
    """
    logger.info(f"Selecting most correct image from {len(results)} search results")
    if not results:
        logger.info("No search results provided to select from")
        return []
    
    # Step 1: Filter out placeholders, empty image URLs, and typical UI graphics (logos, icons)
    valid_candidates = []
    for item in results:
        url = item.get("image_url") or ""
        title = item.get("title") or ""
        
        if not url or "placeholder.svg" in url:
            continue
            
        url_lower = url.lower()
        title_lower = title.lower()
        if any(x in url_lower or x in title_lower for x in ['logo', 'icon', 'avatar', 'sprite']):
            logger.debug(f"Filtering out potential UI graphic: {url}")
            continue
            
        valid_candidates.append(item)
        
    if valid_candidates:
        logger.info(f"Selected primary valid candidate: {valid_candidates[0]}")
        return [valid_candidates[0]]
        
    # Step 2: Fallback to any result with a real image URL
    for item in results:
        url = item.get("image_url") or ""
        if url and "placeholder.svg" not in url:
            logger.info(f"Fallback selected image with valid URL: {item}")
            return [item]
            
    # Step 3: Absolute fallback to the first result
    logger.info(f"Absolute fallback selected image (first result): {results[0]}")
    return [results[0]]

@app.route("/")
def index():
    has_api_key = GEMINI_API_KEY is not None
    return render_template("index.html", has_api_key=has_api_key)

@app.route("/api/upload", methods=["POST"])
def upload():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured on the server."}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
        
    try:
        image_bytes = file.read()
        mime_type = file.mimetype or "image/jpeg"
        
        # Call the multimodal extractor ADK Agent
        raw_result = run_extractor_agent(image_bytes, mime_type)
        logger.info(f"Extractor Agent raw result: {raw_result}")
        
        products = extract_json_array(raw_result)
        return jsonify({
            "success": True,
            "products": products
        })
    except Exception as e:
        logger.exception("Error during upload processing:")
        return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json or {}
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    # Check if Google Custom Search API is configured
    search_key = os.environ.get("GOOGLE_SEARCH_KEY")
    search_cx = os.environ.get("GOOGLE_SEARCH_CX")
    
    if search_key and search_cx:
        logger.info(f"Using Google Custom Search API for query: {query}")
        results = google_custom_image_search(query, search_key, search_cx)
        method = "Google Custom Search API"
    else:
        logger.info(f"Using Google Search Grounding fallback for query: {query}")
        results = google_grounding_image_search(query, GEMINI_API_KEY)
        method = "Google Search Grounding (Fallback)"
        
    # Filter to keep only the single most correct image
    best_results = select_most_correct_image(results)
    
    return jsonify({
        "success": True,
        "results": best_results,
        "method": method
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    session_id = data.get("sessionId", "default_chat_session").strip()
    scanned_items = data.get("scannedItems", [])
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
        
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured."}), 500
        
    try:
        # Call the Chat Assistant ADK Agent
        response_text = run_chat_agent(message, session_id, scanned_items)
        return jsonify({
            "success": True,
            "response": response_text
        })
    except Exception as e:
        logger.exception("Error during chat assistant invocation:")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
