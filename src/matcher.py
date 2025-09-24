import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from collections import defaultdict
from .llm_search import LLMSearchAgent
from .screenshotter import ProductScreenshotter

logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(self):
        self.search_agent = LLMSearchAgent()
    
    async def _find_url_for_item(self, item: Dict) -> tuple[int, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Find URL, price, and screenshot data for a single item (async)."""
        item_number = item.get('item_number', 0)
        try:
            # LLM handles everything including screenshots
            url, price = await self.search_agent.find_product_url_async(item)
            
            # Extract screenshot data if URL was processed
            screenshot_data = None
            if url and url not in ["NOT_FOUND", "ERROR"]:
                screenshot_data = self.search_agent.get_screenshot_data_for_url(url)
            
            return item_number, url, price, screenshot_data
        except Exception as e:
            logger.error(f"Error processing item {item_number}: {e}")
            return item_number, None, None, None

    async def process_items_with_worker_pool(self, items: List[Dict], limit: Optional[int] = None) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """
        Process items with parallel LLM calls using a worker pool.
        LLM handles everything including screenshots.
        """
        logger.info("🚀 Starting product search pipeline")
        
        if limit:
            items = items[:limit]
            logger.info(f"Limited to first {limit} items")
        
        try:
            # Run all LLM calls in parallel - each will handle its own screenshots
            logger.info(f"Starting {len(items)} parallel LLM search tasks...")
            llm_tasks = [self._find_url_for_item(item) for item in items]
            llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
            
            # Process results
            url_mapping = {}
            price_mapping = {}
            screenshot_mapping = {}
            
            for i, result in enumerate(llm_results):
                if isinstance(result, Exception):
                    item_number = items[i].get('item_number', i)
                    logger.error(f"LLM task failed for item {item_number}: {result}")
                    url_mapping[item_number] = "ERROR"
                    price_mapping[item_number] = ""
                    screenshot_mapping[item_number] = {}
                else:
                    item_number, url, price, screenshot_data = result
                    url_mapping[item_number] = url or "NOT_FOUND"
                    price_mapping[item_number] = price or ""
                    screenshot_mapping[item_number] = screenshot_data or {}
            
            # Summary
            found_count = len([url for url in url_mapping.values() if url and url not in ["NOT_FOUND", "ERROR"]])
            price_count = len([price for price in price_mapping.values() if price])
            screenshot_count = len([data for data in screenshot_mapping.values() if data.get('screenshot_path')])
            
            logger.info("🎉 Pipeline complete!")
            logger.info(f"✅ Found {found_count}/{len(items)} products ({price_count} with prices, {screenshot_count} screenshots)")
            
            return url_mapping, price_mapping, screenshot_mapping
            
        finally:
            # Cleanup LLM resources
            await self.search_agent.cleanup_screenshotter()

# Cleanup is now handled by the LLM search agent

    # Keep existing methods for backward compatibility
    async def find_urls_for_items(self, items: List[Dict], limit: Optional[int] = None) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, Any]]]:
        """Find URLs and extract prices for a list of product items using async parallel processing."""
        logger.info(f"🔍 Processing {len(items)} products (async pipeline)")
        
        if limit:
            items = items[:limit]
            logger.info(f"Limited to first {limit} items")
        
        # Create tasks for all items to run in parallel
        tasks = [self._find_url_for_item_legacy(item) for item in items]
        
        # Execute all tasks concurrently
        logger.info(f"Starting {len(tasks)} parallel LLM search tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        url_mapping = {}
        price_mapping = {}
        
        # Process results
        screenshot_mapping = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                item_number = items[i].get('item_number', i)
                logger.error(f"Task failed for item {item_number}: {result}")
                url_mapping[item_number] = "ERROR"
                price_mapping[item_number] = ""
                screenshot_mapping[item_number] = {}
            else:
                item_number, url, price, screenshot_data = result
                url_mapping[item_number] = url or "NOT_FOUND"
                price_mapping[item_number] = price or ""
                screenshot_mapping[item_number] = screenshot_data or {}
                
                # Log progress
                if url:
                    logger.info(f"✓ Found URL for item {item_number}: {price}")
                else:
                    logger.warning(f"✗ No URL found for item {item_number}")
        
        # Summary
        found_count = len([url for url in url_mapping.values() if url and url not in ["NOT_FOUND", "ERROR"]])
        price_count = len([price for price in price_mapping.values() if price])
        screenshot_count = len([data for data in screenshot_mapping.values() if data])
        logger.info(f"✅ Async search complete: {found_count}/{len(items)} found ({price_count} prices, {screenshot_count} screenshots)")
        
        return url_mapping, price_mapping, screenshot_mapping

    async def _find_url_for_item_legacy(self, item: Dict) -> tuple[int, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Legacy method that includes screenshots - for backward compatibility."""
        item_number = item.get('item_number', 0)
        try:
            url, price = await self.search_agent.find_product_url_async(item)
            
            # Extract screenshot data if URL was processed
            screenshot_data = None
            if url and url not in ["NOT_FOUND", "ERROR"]:
                screenshot_data = self.search_agent.get_screenshot_data_for_url(url)
                
                # If LLM didn't take a screenshot during search, take one now
                if not screenshot_data:
                    logger.info(f"📸 Taking screenshot for item {item_number}")
                    try:
                        screenshot_result = await self.search_agent.take_screenshot(url)
                        if screenshot_result.get("success"):
                            screenshot_data = self.search_agent.get_screenshot_data_for_url(url)
                            logger.info(f"📸 Screenshot saved for item {item_number}")
                        else:
                            logger.warning(f"Screenshot failed for item {item_number}: {screenshot_result.get('error', 'Unknown error')}")
                            screenshot_data = {
                                'screenshot_path': '',
                                'price': '',
                                'title': '',
                                'error': screenshot_result.get('error', 'Screenshot failed')
                            }
                    except Exception as e:
                        logger.error(f"Error taking screenshot for item {item_number}: {e}")
                        screenshot_data = {
                            'screenshot_path': '',
                            'price': '',
                            'title': '',
                            'error': f'Screenshot error: {str(e)}'
                        }
            
            return item_number, url, price, screenshot_data
        except Exception as e:
            logger.error(f"Error processing item {item_number}: {e}")
            return item_number, None, None, None

    async def process_items_streaming(self, items: List[Dict], limit: Optional[int] = None) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """
        Complete processing pipeline with streaming: URLs are processed immediately 
        as they come back from LLM calls, without waiting for all LLM calls to finish.
        """
        logger.info("Starting streaming product matching pipeline")
        
        if limit:
            items = items[:limit]
            logger.info(f"Limited to first {limit} items")
        
        # No need to initialize screenshotter - LLM handles screenshots directly
        
        try:
            # Create LLM tasks that will handle everything including screenshots
            async def llm_task_with_streaming(item: Dict):
                """LLM task that handles URL finding and screenshots."""
                item_number, url, price, screenshot_data = await self._find_url_for_item(item)
                
                # Store screenshot data if available
                if screenshot_data:
                    self.screenshot_results[item_number] = screenshot_data
                
                return item_number, url, price, screenshot_data
            
            # Start all LLM tasks
            logger.info(f"Starting {len(items)} parallel LLM search tasks with streaming...")
            llm_tasks = [llm_task_with_streaming(item) for item in items]
            
            # Execute LLM tasks and wait for completion
            results = await asyncio.gather(*llm_tasks, return_exceptions=True)
            
            # Process LLM results
            url_mapping = {}
            price_mapping = {}
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    item_number = items[i].get('item_number', i)
                    logger.error(f"LLM task failed for item {item_number}: {result}")
                    url_mapping[item_number] = "ERROR"
                    price_mapping[item_number] = ""
                    self.screenshot_results[item_number] = {}
                else:
                    item_number, url, price, screenshot_data = result
                    url_mapping[item_number] = url or "NOT_FOUND"
                    price_mapping[item_number] = price or ""
                    # Screenshot data is already stored in llm_task_with_streaming
            
            # LLM handles everything now, no need for domain processors
            
            # Summary
            found_count = len([url for url in url_mapping.values() if url and url not in ["NOT_FOUND", "ERROR"]])
            price_count = len([price for price in price_mapping.values() if price])
            screenshot_count = len([data for data in self.screenshot_results.values() if data.get('screenshot_path')])
            
            logger.info("=== STREAMING PIPELINE COMPLETE ===")
            logger.info(f"URLs found: {found_count}/{len(items)}")
            logger.info(f"LLM-extracted prices: {price_count}")
            logger.info(f"Screenshots captured: {screenshot_count}")
            
            return url_mapping, price_mapping, self.screenshot_results.copy()
            
        finally:
            # Clean up LLM screenshotter if it was initialized
            await self.search_agent.cleanup_screenshotter()
            # Reset screenshot results for next run
            self.screenshot_results.clear()

    async def capture_screenshots_for_urls(self, url_mapping: Dict[int, str]) -> Dict[int, Dict[str, str]]:
        """Capture screenshots and extract data for found URLs using domain-based async queues."""
        # Filter out items without valid URLs
        valid_urls = {k: v for k, v in url_mapping.items() 
                     if v and v not in ["NOT_FOUND", "ERROR"]}
        
        if not valid_urls:
            logger.warning("No valid URLs to process for screenshots")
            return {}
        
        logger.info(f"Capturing screenshots for {len(valid_urls)} URLs using domain-based async queues")
        
        # Use the async screenshotter with domain-based queues
        async with ProductScreenshotter() as screenshotter:
            screenshot_data = await screenshotter.process_urls(valid_urls)
        
        # Summary
        success_count = len([data for data in screenshot_data.values() 
                           if data.get('screenshot_path')])
        logger.info(f"Screenshot capture complete: {success_count}/{len(valid_urls)} screenshots captured")
        
        return screenshot_data
    
    async def process_items_async(self, items: List[Dict], limit: Optional[int] = None) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """Complete processing pipeline with async LLM calls (LLM handles screenshots directly now)."""
        logger.info("Starting async product matching pipeline")
        
        # LLM now handles everything including screenshots
        logger.info("Processing URLs, prices, and screenshots using parallel async LLM calls...")
        url_mapping, price_mapping, screenshot_data = await self.find_urls_for_items(items, limit)
        
        # Clean up LLM screenshotter
        await self.search_agent.cleanup_screenshotter()
        
        logger.info("Async product matching pipeline complete")
        return url_mapping, price_mapping, screenshot_data
    
    async def process_single_item_async(self, item: Dict) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """Process a single item and return the results."""
        logger.info(f"Processing single item {item.get('item_number')}")
        
        try:
            # Get URL, price, and screenshot data for the item (LLM handles everything)
            item_number, url, price, screenshot_data = await self._find_url_for_item(item)
            
            url_mapping = {item_number: url or "NOT_FOUND"}
            price_mapping = {item_number: price or ""}
            screenshot_mapping = {item_number: screenshot_data or {}}
            
            return url_mapping, price_mapping, screenshot_mapping
            
        finally:
            # Clean up LLM screenshotter
            await self.search_agent.cleanup_screenshotter()

    def process_single_item(self, item: Dict) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """Synchronous wrapper for processing a single item."""
        return asyncio.run(self.process_single_item_async(item))

    def process_items(self, items: List[Dict], limit: Optional[int] = None, use_worker_pool: bool = True) -> tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, str]]]:
        """
        Complete processing pipeline (synchronous wrapper for main.py compatibility).
        
        Args:
            items: List of product items to process
            limit: Optional limit on number of items to process
            use_worker_pool: If True, use worker pool pipeline (parallel LLM + direct screenshots).
                            If False, use legacy streaming pipeline.
        """
        if use_worker_pool:
            logger.info("Starting worker pool pipeline with parallel LLM calls")
            return asyncio.run(self.process_items_with_worker_pool(items, limit))
        else:
            logger.info("Starting legacy streaming product matching pipeline")
            return asyncio.run(self.process_items_streaming(items, limit)) 