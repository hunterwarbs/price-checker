import json
import logging
import time
import asyncio
import hashlib
import random
import os
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI

from .config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 
    DEFAULT_MODEL, GOOGLE_SEARCH_LIMIT, MAX_TOOL_CALLS_PER_ITEM,
    SEARCH_SYSTEM_PROMPT, SEARCH_USER_PROMPT_TEMPLATE, TEMPERATURE,
    OPENROUTER_MAX_REQUESTS_PER_MINUTE,
    OXYLABS_WEB_API_USERNAME, OXYLABS_WEB_API_PASSWORD,
    FIREWORKS_API_KEY, FIREWORKS_BASE_URL
)
from .smart_rate_limiter import global_rate_manager, ProviderType
from urllib.parse import urlparse
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Old rate limiters removed - now using smart adaptive rate limiter

def _assistant_message_dict(message_obj: Any) -> Dict[str, Any]:
    """Return a sanitized assistant message with only role and content."""
    content = getattr(message_obj, "content", None)
    return {"role": "assistant", "content": content or ""}

class LLMSearchAgent:
    def __init__(self):
        # Initialize Fireworks (OpenAI-compatible) client
        import httpx
        self.openai_client = OpenAI(
            base_url=FIREWORKS_BASE_URL,
            api_key=FIREWORKS_API_KEY,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=500, max_keepalive_connections=200)
            )
        )

        # Dedicated thread pool for blocking OpenAI-compatible calls
        max_workers = int(os.getenv("OPENROUTER_THREAD_POOL", "100"))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Fireworks text client thread pool initialised with {max_workers} workers")
        
        # Smart rate limiters are now managed globally
        # Access via global_rate_manager.execute_*_request() methods
        
        # Screenshot data storage
        self.screenshot_data = {}
        self.screenshotter = None
        
        # Current item being processed (for logging context)
        self.current_item_number = None
        
        # Shared screenshotter instance (reuse to avoid re-initializing OCR)
        self._shared_screenshotter = None
        
        # Define tools for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for product information using AI-powered search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding product information"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of search results to return (default: 5, max: 10)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Take a screenshot of a webpage and extract all visible text using OCR. Returns the full OCR text that you can analyze to find product information, prices, titles, etc.",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the webpage to screenshot"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    async def _get_shared_screenshotter(self):
        """Get or create shared screenshotter instance (reuses OCR and contexts per domain)."""
        if self._shared_screenshotter is None:
            from .screenshotter import ProductScreenshotter
            self._shared_screenshotter = ProductScreenshotter()
            await self._shared_screenshotter.__aenter__()
        return self._shared_screenshotter

    async def _take_screenshot_isolated(self, url: str, item_number: int) -> Dict[str, Any]:
        """Take a screenshot reusing per-domain context (rotates after threshold)."""
        try:
            item_prefix = f"Item {item_number}: "
            logger.info(f"{item_prefix}📸 Taking screenshot (reuse domain context)")
            
            # Execute screenshot with smart Oxylabs rate limiting
            async def oxylabs_screenshot_request():
                # Get shared screenshotter (reuses OCR and domain context)
                screenshotter = await self._get_shared_screenshotter()
                # Use domain-reuse path: create or reuse a context internally per domain
                # Map to existing API by delegating to process of a single URL via reuse helper
                return await screenshotter.capture_product_page_on_existing_page(
                    # Create or obtain a page by opening a short-lived context for single-domain reuse
                    # We will let screenshotter handle creating a context if needed by calling capture_product_page
                    # but prefer reuse by routing through the domain queue helper for a single item
                    # Fallback to direct if reuse is not available
                    # Simpler approach: call capture_product_page (it already uses current behavior),
                    # but we changed _process_domain_queue to reuse contexts; to ensure reuse here, call capture_product_page
                    # and rely on per-domain queues in matcher for batch; for single calls, just use capture_product_page
                    # so behavior is consistent
                    # However, to avoid requiring an existing page object, just call capture_product_page
                    # The function name remains, but we delegate to capture_product_page
                    # Note: This preserves screenshot fidelity
                    #
                    # Since capture_product_page_on_existing_page requires a Page, instead call capture_product_page
                    # directly here:
                )
            
            # Use direct capture with internal reuse via screenshotter where applicable
            async def oxylabs_screenshot_request_direct():
                screenshotter = await self._get_shared_screenshotter()
                return await screenshotter.capture_product_page(url, item_number)

            result = await global_rate_manager.execute_oxylabs_proxy_request(oxylabs_screenshot_request_direct)
            
            # Store the result
            self.screenshot_data[url] = result
            
            item_prefix = f"Item {item_number}: "
            logger.info(f"{item_prefix}✅ Screenshot completed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to take isolated screenshot for {url}: {e}")
            error_result = {
                'screenshot_path': '',
                'ocr_text': '',
                'error': str(e)
            }
            self.screenshot_data[url] = error_result
            return error_result

    async def search_web_async(self, query: str, num_results: int = GOOGLE_SEARCH_LIMIT) -> Dict[str, Any]:
        """Google search using Oxylabs Web Scraper API - gives LLM flexible retailer choice."""
        try:
            item_prefix = f"Item {self.current_item_number}: " if self.current_item_number else ""
            logger.info(f"{item_prefix}🔍 Searching: {query[:60]}{'...' if len(query) > 60 else ''}")
            
            # Execute Google search with smart rate limiting
            async def google_search_request():
                import requests
                payload = {
                    "source": "google_search",
                    "query": query,
                    "parse": True,
                    "limit": num_results,
                    "geo_location": "United States"
                }

                # Run the blocking requests.post in a dedicated thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,
                    lambda: requests.post(
                        "https://realtime.oxylabs.io/v1/queries",
                        auth=(OXYLABS_WEB_API_USERNAME, OXYLABS_WEB_API_PASSWORD),
                        json=payload,
                        timeout=30
                    )
                )
                response.raise_for_status()
                return response.json()
            
            oxylabs_response = await global_rate_manager.execute_oxylabs_web_api_request(google_search_request)
            
            # Parse Oxylabs Google search results
            formatted_results = []
            
            if oxylabs_response.get("results") and oxylabs_response["results"][0].get("content", {}).get("results"):
                content_results = oxylabs_response["results"][0]["content"]["results"]
                
                # Extract organic results (main search results)
                organic_results = content_results.get("organic", [])
                for result in organic_results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("desc", ""),
                        "url_shown": result.get("url_shown", ""),
                        "position": result.get("pos", 0),
                        "source": "organic"
                    })
                
                # Also include Google Shopping results if available
                shopping_results = content_results.get("shopping", []) 
                for result in shopping_results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": f"Price: {result.get('price', 'N/A')} from {result.get('seller', 'Unknown')}",
                        "price": result.get("price"),
                        "seller": result.get("seller"),
                        "position": result.get("pos", 0),
                        "source": "shopping"
                    })
            
            item_prefix = f"Item {self.current_item_number}: " if self.current_item_number else ""
            logger.info(f"{item_prefix}🔍 Found {len(formatted_results)} search results")
            return {
                "results": formatted_results[:num_results],
                "query": query,
                "search_type": "google",
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in Google search: {e}")
            return {"results": [], "error": str(e)}

    def search_web(self, query: str, num_results: int = GOOGLE_SEARCH_LIMIT) -> Dict[str, Any]:
        """Search the web (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.search_web_async(query, num_results))
                    return future.result()
            else:
                return asyncio.run(self.search_web_async(query, num_results))
        except Exception as e:
            logger.error(f"❌ Error in sync search wrapper: {e}")
            return {"results": [], "error": str(e)}
        
    async def take_screenshot(self, url: str, item_number: Optional[int] = None) -> Dict[str, Any]:
        """Take a screenshot with isolated connection - no caching, fresh IP every time."""
        if item_number is None:
            item_number = getattr(self, 'current_item_number', '?')
        item_prefix = f"Item {item_number}: "
        logger.info(f"{item_prefix}📸 Taking fresh screenshot")
        return await self._take_screenshot_isolated(url, item_number)
    
    async def _execute_function_call_async(self, function_name: str, arguments: Dict) -> Dict[str, Any]:
        """Async version of execute_function_call."""
        if function_name == "search_web":
            return await self.search_web_async(arguments.get("query", ""), arguments.get("num_results", 5))
        elif function_name == "take_screenshot":
            item_number = arguments.get("item_number", self.current_item_number)
            return await self.take_screenshot(arguments.get("url", ""), item_number)
        else:
            return {"error": f"Unknown function: {function_name}"}

    def _execute_function_call(self, function_name: str, arguments: Dict) -> Dict[str, Any]:
        """Execute a function call from the LLM."""
        if function_name == "search_web":
            return self.search_web(arguments.get("query", ""), arguments.get("num_results", 5))
        elif function_name == "take_screenshot":
            arguments_with_item = arguments.copy()
            arguments_with_item["item_number"] = self.current_item_number
            return {"_async_call": True, "function": "take_screenshot", "arguments": arguments_with_item}
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    async def find_product_url_async(self, item: Dict[str, Any], max_retries: int = 3) -> tuple[Optional[str], Optional[str]]:
        """Async version of find_product_url for parallel processing."""
        
        self.current_item_number = item.get('item_number', '?')
        
        for attempt in range(max_retries):
            try:
                user_prompt = SEARCH_USER_PROMPT_TEMPLATE.format(
                    make=item.get('make', ''),
                    model=item.get('model', ''),
                    description=item.get('description', ''),
                    room=item.get('room', ''),
                    original_vendor=item.get('original_vendor', ''),
                    quantity_lost=item.get('quantity_lost', 1),
                    item_age_months=item.get('item_age_months', 'Unknown'),
                    cost_to_replace_each=item.get('cost_to_replace_each', 0),
                    total_cost=item.get('total_cost', 0),
                    price=item.get('price', 0)
                )
                
                item_num = item.get('item_number', '?')
                desc = item.get('description', '')[:30]
                logger.info(f"🛍️ Looking for item {item_num}: {desc}{'...' if len(item.get('description', '')) > 30 else ''}")
                
                messages = [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                async def fireworks_request():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self._executor,
                        lambda: self.openai_client.chat.completions.create(
                            model=DEFAULT_MODEL,
                            messages=messages,
                            tools=self.tools,
                            tool_choice="auto",
                            temperature=TEMPERATURE
                        )
                    )
                
                response = await global_rate_manager.execute_openrouter_request(fireworks_request)
                
                message = response.choices[0].message
                messages.append(_assistant_message_dict(message))
                
                tool_call_count = 0
                while message.tool_calls and tool_call_count < MAX_TOOL_CALLS_PER_ITEM:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        logger.debug(f"Item {self.current_item_number}: LLM calling: {function_name}")
                        function_result = await self._execute_function_call_async(function_name, arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(function_result)
                        })
                        tool_call_count += 1
                    
                    async def followup_fireworks_request():
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            self._executor,
                            lambda: self.openai_client.chat.completions.create(
                                model=DEFAULT_MODEL,
                                messages=messages,
                                tools=self.tools,
                                tool_choice="auto",
                                temperature=TEMPERATURE
                            )
                        )
                    response = await global_rate_manager.execute_openrouter_request(followup_fireworks_request)
                    message = response.choices[0].message
                    messages.append(_assistant_message_dict(message))
                
                final_response = message.content
                if final_response:
                    try:
                        result = json.loads(final_response)
                        url = result.get('url', '')
                        price = result.get('price', '')
                        if url and url not in ['NOT_FOUND', 'ERROR']:
                            logger.info(f"✅ Item {item.get('item_number', '?')}: Found at ${price}")
                            return url, price
                        else:
                            logger.warning(f"❌ Item {item.get('item_number', '?')}: Not found")
                            return None, None
                    except json.JSONDecodeError:
                        import re
                        url_pattern = r'https?://[^\s<>"]+'
                        urls = re.findall(url_pattern, final_response)
                        if urls:
                            url = urls[0].rstrip('.,!?;')
                            logger.info(f"🔄 Item {item.get('item_number', '?')}: Found (fallback)")
                            return url, None
                        if final_response.startswith('http'):
                            url = final_response.strip().rstrip('.,!?;')
                            logger.info(f"🔄 Item {item.get('item_number', '?')}: Found (fallback)")
                            return url, None
                
                logger.warning(f"⚠️ Item {item.get('item_number', '?')}: No URL in response")
                return None, None
                
            except Exception as e:
                logger.error(f"Error in find_product_url_async (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    self.current_item_number = None
                    return None, None
        
        self.current_item_number = None
        return None, None

    def find_product_url(self, item: Dict[str, Any], max_retries: int = 3) -> tuple[Optional[str], Optional[str]]:
        """Sync wrapper for product URL discovery."""
        self.current_item_number = item.get('item_number', '?')
        for attempt in range(max_retries):
            try:
                user_prompt = SEARCH_USER_PROMPT_TEMPLATE.format(
                    make=item.get('make', ''),
                    model=item.get('model', ''),
                    description=item.get('description', ''),
                    room=item.get('room', ''),
                    original_vendor=item.get('original_vendor', ''),
                    quantity_lost=item.get('quantity_lost', 1),
                    item_age_months=item.get('item_age_months', 'Unknown'),
                    cost_to_replace_each=item.get('cost_to_replace_each', 0),
                    total_cost=item.get('total_cost', 0),
                    price=item.get('price', 0)
                )
                
                messages = [
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.openai_client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=TEMPERATURE
                )
                
                message = response.choices[0].message
                messages.append(_assistant_message_dict(message))
                
                tool_call_count = 0
                while message.tool_calls and tool_call_count < MAX_TOOL_CALLS_PER_ITEM:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        logger.debug(f"Item {self.current_item_number}: LLM calling: {function_name}")
                        if function_name == "take_screenshot":
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                                function_result = loop.run_until_complete(
                                    self.take_screenshot(arguments.get("url", ""), self.current_item_number)
                                )
                            except Exception:
                                function_result = asyncio.run(
                                    self.take_screenshot(arguments.get("url", ""), self.current_item_number)
                                )
                        else:
                            function_result = self._execute_function_call(function_name, arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(function_result)
                        })
                        tool_call_count += 1
                    
                    response = self.openai_client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=TEMPERATURE
                    )
                    message = response.choices[0].message
                    messages.append(_assistant_message_dict(message))
                
                final_response = message.content
                if final_response:
                    try:
                        result = json.loads(final_response)
                        url = result.get('url', '')
                        price = result.get('price', '')
                        if url and url not in ['NOT_FOUND', 'ERROR']:
                            logger.info(f"✅ Item {item.get('item_number', '?')}: Found at ${price}")
                            return url, price
                        else:
                            logger.warning(f"❌ Item {item.get('item_number', '?')}: Not found")
                            return None, None
                    except json.JSONDecodeError:
                        import re
                        url_pattern = r'https?://[^\s<>"]+'
                        urls = re.findall(url_pattern, final_response)
                        if urls:
                            url = urls[0].rstrip('.,!?;')
                            logger.info(f"🔄 Item {item.get('item_number', '?')}: Found (fallback)")
                            return url, None
                        if final_response.startswith('http'):
                            url = final_response.strip().rstrip('.,!?;')
                            logger.info(f"🔄 Item {item.get('item_number', '?')}: Found (fallback)")
                            return url, None
                
                logger.warning(f"⚠️ Item {item.get('item_number', '?')}: No URL in response")
                return None, None
                
            except Exception as e:
                logger.error(f"Error in find_product_url (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None, None
        
        return None, None
    
    def get_screenshot_data_for_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get screenshot data for a specific URL."""
        return self.screenshot_data.get(url)
    
    async def cleanup_screenshotter(self):
        """Clean up screenshot data and shared screenshotter instance."""
        try:
            self.screenshot_data.clear()
            logger.info("Screenshot data cleared")
            if self._shared_screenshotter:
                try:
                    await self._shared_screenshotter.__aexit__(None, None, None)
                    logger.info("🧹 Cleaned up shared screenshotter resources")
                except Exception as e:
                    logger.error(f"Error cleaning up shared screenshotter: {e}")
                finally:
                    self._shared_screenshotter = None
        except Exception as e:
            logger.error(f"Error during LLM screenshot cleanup: {e}")