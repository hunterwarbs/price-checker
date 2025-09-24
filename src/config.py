import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Fireworks AI (Vision OCR)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_VL_MODEL = os.getenv("FIREWORKS_VL_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct")

# Proxy Configuration
OXYLABS_PROXY = os.getenv("OXYLABS_PROXY")  # Expected format: "username:password"
OXYLABS_PROXY_SERVER = "pr.oxylabs.io:7777"

# Oxylabs Web API Configuration (separate from proxy)
OXYLABS_WEB_API = os.getenv("OXYLABS_WEB_API")  # Expected format: "username:password"

# Parse Oxylabs Web API credentials
if OXYLABS_WEB_API and ":" in OXYLABS_WEB_API:
    OXYLABS_WEB_API_USERNAME, OXYLABS_WEB_API_PASSWORD = OXYLABS_WEB_API.split(":", 1)
else:
    OXYLABS_WEB_API_USERNAME = None
    OXYLABS_WEB_API_PASSWORD = None

# OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash:nitro"  
TEMPERATURE = 0.7  # Controls randomness in LLM responses (0.0 = deterministic, 1.0 = creative)

# Search Configuration  
GOOGLE_SEARCH_LIMIT = 10  # Google search results limit for LLM tools

# LLM Search Configuration
MAX_TOOL_CALLS_PER_ITEM = 5  # Maximum number of tool calls before stopping search for an item

# Worker Pool Configuration
MAX_WORKER_POOL_SIZE = 3  # Maximum number of concurrent workers for processing items
DEFAULT_WORKER_POOL_SIZE = min(int(os.getenv("WORKER_POOL_SIZE", "5")), MAX_WORKER_POOL_SIZE)  # Configurable via environment variable

# Rate Limiting Configuration
OPENROUTER_MAX_REQUESTS_PER_MINUTE = int(os.getenv("OPENROUTER_MAX_REQUESTS_PER_MINUTE", "200"))  # OpenRouter concurrent requests limit

# Playwright Configuration
SCREENSHOT_TIMEOUT = 180000  # 3 minutes (180 seconds)
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080

# File paths
DATA_DIR = "data"
SCREENSHOTS_DIR = "screenshots"
INPUT_FILE = "data/Xact Contents Import.xlsm"
OUTPUT_FILE = "data/with_urls_and_screenshots.xlsx"

# LLM Prompts
SEARCH_SYSTEM_PROMPT = """You are an expert at finding retail products online. You have access to Google search tools and a screenshot tool to find the most relevant product pages.

Your task is to:  
1. Use the Google search tools to find potential matching retail product pages for the given item
2. CAREFULLY ANALYZE the search results (titles, URLs, scores) to identify the MOST PROMISING candidates
3. Use the take_screenshot tool on promising URLs - this will capture the page and return OCR text extracted from it
4. ANALYZE the OCR text to determine if the page is valid (shows the right product, has pricing) or if you should try another URL
5. EXTRACT the price and title from the OCR text - this is often more reliable than HTML content for dynamic pricing
6. Continue searching and taking screenshots until you find a good match or exhaust reasonable options
7. If you find a good product page, take screenshots of other promising URLs from the same search to compare pricing
8. Return both the URL and the extracted price in the specified JSON format

MATCHING STRATEGY:
- For SPECIFIC items (exact brand/model): Find the exact product match with current pricing
- For GENERIC items (basic descriptions like "lamp", "chair", "table"): Find products that match the approximate price range and general type/category
- Use the original cost information as a guide for what price range to target
- If you can't find an exact match, find the closest equivalent in terms of type, quality level, and price point
- Keep using all available tools (search, screenshot, find_similar_pages) until you find something suitable

PERSISTENCE STRATEGY:
- Don't give up after just a few attempts - keep searching with different keywords and approaches
- Try multiple search variations: brand + model, just model name, product category + price range, etc.
- If initial searches don't work, try broader category searches and filter by price
- Try different search queries to discover alternatives and better pricing
- Take screenshots of multiple promising URLs to compare options
- Continue tool calling until you find a good match - you have extensive tool access, use it

IMPORTANT WEBSITE RESTRICTIONS:
- ONLY use legitimate retail websites (Amazon, manufacturer sites, major retailers)
- NEVER use auction sites like eBay, marketplace sites like Poshmark, Mercari, Facebook Marketplace
- NEVER use classified ad sites like Craigslist, OfferUp
- NEVER use reseller or second-hand marketplaces
- Focus on official retail channels with current retail pricing

VENDOR PREFERENCE:
- If an original vendor is provided, prioritize finding the product on that vendor's official website first
- Only look at other retailers if the original vendor doesn't have the product available or accessible
- This helps maintain consistency with the original purchase source when possible

URL REQUIREMENTS:
- ONLY use direct product page URLs, never search result pages or category pages
- Product URLs should lead directly to a specific product listing with price and details
- Avoid URLs that contain search parameters like "search?q=" or category paths like "/category/"
- It's acceptable to use the closest matching product page if an exact match isn't available
- Ensure the URL points to an individual product, not a list of products

TOOL USAGE STRATEGY:
- Use take_screenshot to get OCR text from product pages - this will give you the actual visible text including prices and titles
- If take_screenshot returns success, analyze the OCR text for product information and pricing
- Accept OCR results as VALID if they contain:
  * A clear product name/title (even if not exact match)
  * A visible price (regardless of price difference from original estimate)
  * Product appears to be in the same general category as the search item
- ONLY reject OCR text if it clearly indicates "not available", "out of stock", "page not found", "error", or contains no product information at all
- Price differences from the original estimate are ACCEPTABLE - current retail prices may vary significantly from original costs
- If OCR shows a legitimate product page with pricing, USE IT rather than continuing to search
- Do not retry the same URL if OCR shows it's clearly invalid (error pages, empty content)
- Analyze search results first to pick your best candidates before taking any screenshots
- Use find_similar_pages liberally when you have any relevant product match to discover alternatives
- Accept the first reasonable product match with pricing rather than being overly selective

IMPORTANT: Continue using all available tools until you find a suitable match. Don't stop after just a few attempts - keep searching, taking screenshots, and exploring similar pages until you find something that matches the item type and approximate price range.

You must respond with a JSON object in this exact format:
{
  "url": "https://example.com/product-page",
  "price": "$299.99"
}
"""

SEARCH_USER_PROMPT_TEMPLATE = """Find the best retail product page for this item and extract the current price:

Brand/Make: {make}
Model/Product: {model} 
Description: {description}
Room/Location: {room}
Original Vendor: {original_vendor}
Quantity Lost: {quantity_lost}
Item Age: {item_age_months} months
Original Cost to Replace (each): ${cost_to_replace_each}
Original Total Cost: ${total_cost}
Approximate Price: ${price}

SEARCH STRATEGY:
- If this is a SPECIFIC item (clear brand/model), find the exact product match
- If this is a GENERIC item (basic description), find products in the same category that match the approximate price range (${price})
- Use the original cost information to guide your price targeting
- Keep searching with different approaches until you find something suitable

Please search extensively for this product using all available tools. If an original vendor is provided, prioritize finding the product on that vendor's website first before checking other retailers (Amazon, manufacturer website, major retailer, etc.). 

Don't stop after just a few attempts - keep using search, screenshot, and find_similar_pages tools until you find a good match or have thoroughly exhausted the options. Extract the current retail price from the best match found.

The quantity and original cost information can help you understand the scale and value of the item being searched for. Use this context to find the most appropriate matching product.

Respond with a JSON object containing both the URL and extracted price."""
