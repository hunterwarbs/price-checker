import asyncio
import logging
import re
import random
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from urllib.parse import urlparse
from collections import defaultdict
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright_stealth import Stealth
from .ocr_analyzer import OCRAnalyzer
from .config import SCREENSHOT_TIMEOUT, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, SCREENSHOTS_DIR, OXYLABS_PROXY, OXYLABS_PROXY_SERVER

logger = logging.getLogger(__name__)

class ProductScreenshotter:
    def __init__(self, use_oxylabs_proxy: bool = True):
        self.browser: Optional[Browser] = None
        self.screenshots_dir = Path(SCREENSHOTS_DIR)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Oxylabs proxy configuration
        self.use_oxylabs_proxy = use_oxylabs_proxy and bool(OXYLABS_PROXY)
        self.oxylabs_credentials = OXYLABS_PROXY
        self.oxylabs_server = OXYLABS_PROXY_SERVER
        
        # Initialize playwright-stealth
        self.stealth = Stealth()
        
        # Initialize OCR analyzer
        self.ocr_analyzer = OCRAnalyzer()
        
        # Thread pool for OCR processing to avoid blocking
        import concurrent.futures
        import os
        max_ocr_workers = int(os.getenv("OCR_THREAD_POOL", "10"))
        self.ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_ocr_workers)
        logger.info(f"OCR thread pool initialized with {max_ocr_workers} workers")
        
        proxy_status = "with proxy" if self.use_oxylabs_proxy else "direct"
        logger.info(f"🎭 Screenshot service ready ({proxy_status})")
    
    def _extract_ocr_fast(self, image_path: str, item_number: int) -> str:
        """Fast OCR extraction using simplified approach to avoid timeouts."""
        try:
            # Use fast mode (single preprocessing + single config)
            ocr_text = self.ocr_analyzer.extract_text_from_image(image_path, fast_mode=True)
            
            # Save OCR text to file for debugging
            ocr_txt_path = self.screenshots_dir / f"item_{item_number}.txt"
            try:
                with open(ocr_txt_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
                logger.debug(f"OCR text saved to {ocr_txt_path.name}")
            except Exception as e:
                logger.warning(f"Could not save OCR text: {e}")
            
            return ocr_text
            
        except Exception as e:
            logger.error(f"Fast OCR failed for item {item_number}: {e}")
            return f"OCR Error: {str(e)}"
    
    def _get_oxylabs_proxy_config(self) -> Optional[Dict[str, str]]:
        """Get Oxylabs proxy configuration for Playwright."""
        if not self.use_oxylabs_proxy or not self.oxylabs_credentials:
            return None
        
        # Parse username:password from credentials
        try:
            username, password = self.oxylabs_credentials.split(':', 1)
            # Oxylabs requires http:// prefix for Playwright
            return {
                "server": f"http://{self.oxylabs_server}",
                "username": username,
                "password": password
            }
        except ValueError:
            logger.error("Invalid OXYLABS_PROXY format. Expected 'username:password'")
            return None
    
    async def __aenter__(self):
        """Async context manager entry with playwright-stealth."""
        # Use stealth's built-in playwright management for best stealth configuration
        self.playwright_context_manager = self.stealth.use_async(async_playwright())
        self.playwright = await self.playwright_context_manager.__aenter__()
        
        # Launch browser with basic settings - stealth handles the rest
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright_context_manager'):
            await self.playwright_context_manager.__aexit__(exc_type, exc_val, exc_tb)
        
        # Shutdown OCR thread pool
        if hasattr(self, 'ocr_executor'):
            self.ocr_executor.shutdown(wait=False)
    
    async def _create_stealth_context(self, url: str) -> BrowserContext:
        """Create a browser context with stealth settings and Oxylabs proxy."""
        # Configure Oxylabs proxy if enabled
        proxy_config = self._get_oxylabs_proxy_config()
        if proxy_config:
            logger.debug(f"Using proxy: {proxy_config['username']}@{proxy_config['server']}")
        
        # Create context with basic settings - stealth plugin handles the rest
        context = await self.browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            proxy=proxy_config,
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        # Block unnecessary resources to save bandwidth - keep essential ones for normal appearance
        await context.route("**/*", self._handle_resource_blocking)
        
        # Apply stealth to the context - this handles all the anti-detection measures
        await self.stealth.apply_stealth_async(context)
        
        return context
    
    async def _handle_resource_blocking(self, route, request):
        """Block unnecessary resources to save bandwidth while keeping page appearance normal."""
        resource_type = request.resource_type
        url = request.url.lower()
        
        # Block these resource types that don't affect visual appearance
        if resource_type in ['websocket', 'eventsource', 'beacon', 'ping']:
            # Resource blocked (keeping logs clean)
            await route.abort()
            return
        
        # Block tracking, analytics, and ad resources
        tracking_domains = [
            'google-analytics.com', 'googletagmanager.com', 'doubleclick.net',
            'facebook.com', 'facebook.net', 'googlesyndication.com',
            'googleadservices.com', 'adsystem.com', 'amazon-adsystem.com',
            'scorecardresearch.com', 'quantserve.com', 'hotjar.com',
            'segment.com', 'mixpanel.com', 'amplitude.com', 'intercom.io',
            'zendesk.com', 'zopim.com', 'tawk.to', 'drift.com',
            'outbrain.com', 'taboola.com', 'criteo.com', 'bing.com/th/id',
            'pinterest.com', 'twitter.com', 'linkedin.com', 'instagram.com'
        ]
        
        # Block tracking scripts and pixels
        if any(domain in url for domain in tracking_domains):
            # Blocked tracking (keeping logs clean)
            await route.abort()
            return
        
        # Block known tracking/analytics paths
        if any(path in url for path in ['/analytics', '/tracking', '/gtm.js', '/ga.js', '/fbevents.js', '/pixel']):
            # Blocked tracking path (keeping logs clean)
            await route.abort()
            return
        
        # Allow essential resources for normal appearance
        # - Keep images, stylesheets, fonts, scripts (but not tracking ones)
        # - Keep main document and XHR requests for functionality
        await route.continue_()
    
    async def _human_like_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0) -> None:
        """Add human-like random delays."""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)
    
    async def _simulate_human_behavior(self, page: Page) -> None:
        """Simple human behavior simulation."""
        try:
            # Add some realistic delays and mouse movement
            await self._human_like_delay(0.5, 1.5)
            
            # Simple mouse movement to center of page
            await page.mouse.move(VIEWPORT_WIDTH // 2, VIEWPORT_HEIGHT // 2)
            await self._human_like_delay(0.2, 0.5)
            
            # Optional light scrolling
            if random.random() < 0.3:  # 30% chance
                scroll_amount = random.randint(100, 300)
                await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                await self._human_like_delay(0.5, 1.0)
                
        except Exception as e:
            pass  # Simulation error (non-critical)
    
    async def _wait_for_page_fully_loaded(self, page: Page) -> None:
        """Wait for page to be loaded efficiently - skip networkidle to avoid timeout."""
        try:
            # Wait for DOM and initial resources - this is sufficient for screenshots
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
            await page.wait_for_load_state("load", timeout=15000)
            
            # Skip networkidle - it waits for ALL requests including ads/tracking
            # Instead, wait for essential page content to be visible
            await self._wait_for_content_visible(page)
            
            # Brief wait for any immediate dynamic content
            await self._human_like_delay(1.0, 1.5)
            
        except Exception as e:
            pass  # Page loading wait timeout
    
    async def _wait_for_content_visible(self, page: Page) -> None:
        """Wait for essential page content to be visible."""
        try:
            # Wait for body to have content
            await page.wait_for_function(
                "document.body && document.body.innerText.length > 100",
                timeout=8000
            )
            
            # Wait for images to start loading (but not finish)
            await page.wait_for_function(
                "document.images.length === 0 || Array.from(document.images).some(img => img.complete || img.naturalHeight > 0)",
                timeout=5000
            )
            
        except Exception as e:
            pass  # Content visibility timeout

    async def _handle_page_overlays(self, page: Page) -> None:
        """Efficiently handle cookie consent and popups in one pass."""
        try:
            # Brief wait for any immediate overlays to appear
            await self._human_like_delay(0.5, 1.0)
            
            # Try ESC key first - works for many overlays
            await page.keyboard.press('Escape')
            await asyncio.sleep(0.3)
            
            # Look for cookie consent and close buttons in one pass
            overlay_selectors = [
                # Cookie consent buttons
                'button:has-text("Accept all cookies")', 'button:has-text("Accept All")',
                'button:has-text("Accept")', 'button:has-text("I agree")',
                '[data-testid*="accept"]', '.cookie-accept', '#cookie-accept',
                
                # Common close buttons
                'button[aria-label*="close" i]', 'button[title*="close" i]',
                '[class*="close"]:not(script):not(style)', '.close-button', '.close-btn',
                'button:has-text("×")', 'button:has-text("✕")',
                'button:has-text("No thanks")', 'button:has-text("Skip")',
                'button:has-text("Maybe later")', 'button:has-text("Not now")'
            ]
            
            # Try to click the first visible overlay button
            for selector in overlay_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible() and await element.is_enabled():
                        await element.click(timeout=2000)
                        logger.debug(f"Dismissed overlay: {selector}")
                        await asyncio.sleep(0.5)
                        break
                except:
                    continue
            
            # Scroll to top to ensure consistent screenshot position
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.3)
            
            # Final ESC attempt
            await page.keyboard.press('Escape')
            
        except Exception as e:
            pass  # Overlay handling error (non-critical)

    async def _dismiss_popups(self, page: Page) -> None:
        """Try to dismiss common popups and overlays using multiple robust strategies."""
        dismissed_count = 0
        
        # Strategy 1: Try ESC key first (works for many popups)
        try:
            await page.keyboard.press('Escape')
            await self._human_like_delay(0.5, 1.0)
            logger.debug("Tried ESC key for popup dismissal")
        except:
            pass
        
        # Strategy 2: Look for explicit close buttons with comprehensive selectors
        close_button_selectors = [
            # Common close button patterns
            'button[aria-label*="close" i]',
            'button[aria-label*="dismiss" i]',
            'button[title*="close" i]',
            'button[title*="dismiss" i]',
            '[role="button"][aria-label*="close" i]',
            
            # Class-based selectors
            '[class*="close"]:not(script):not(style)',
            '[class*="dismiss"]:not(script):not(style)',
            '.close-button', '.close-btn', '.btn-close',
            '.modal-close', '.popup-close', '.overlay-close',
            
            # ID-based selectors  
            '#close', '#close-button', '#close-btn',
            '#modal-close', '#popup-close',
            
            # Generic patterns
            '[data-dismiss]', '[data-close]',
            'button:has-text("×")', 'button:has-text("✕")',
            'span:has-text("×")', 'span:has-text("✕")',
            '[class*="modal"] button', '[class*="popup"] button',
            '[class*="overlay"] button', '[class*="dialog"] button',
        ]
        
        for selector in close_button_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if await element.is_visible() and await element.is_enabled():
                        # Check if element looks like a close button
                        text = await element.inner_text()
                        aria_label = await element.get_attribute('aria-label') or ''
                        title = await element.get_attribute('title') or ''
                        class_name = await element.get_attribute('class') or ''
                        
                        # Look for close indicators
                        close_indicators = ['close', 'dismiss', '×', '✕', 'cancel', 'skip']
                        combined_text = f"{text} {aria_label} {title} {class_name}".lower()
                        
                        if any(indicator in combined_text for indicator in close_indicators):
                            await element.click(timeout=2000)
                            await self._human_like_delay(0.3, 0.8)
                            dismissed_count += 1
                            logger.debug(f"Dismissed popup: {selector}")
                            break
            except:
                continue
        
        # Strategy 3: Try text-based button detection
        text_based_selectors = [
            'button:has-text("No thanks")',
            'button:has-text("Not now")',
            'button:has-text("Maybe later")',
            'button:has-text("Skip")',
            'button:has-text("Continue without")',
            'button:has-text("Decline")',
            'button:has-text("Cancel")',
            'button:has-text("Got it")',
            'button:has-text("OK")',
            'button:has-text("Accept")',
            # Common patterns in different languages
            '[class*="decline"]', '[class*="skip"]', '[class*="later"]',
        ]
        
        for selector in text_based_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible() and await element.is_enabled():
                    await element.click(timeout=2000)
                    await self._human_like_delay(0.3, 0.8)
                    dismissed_count += 1
                    logger.debug(f"Dismissed popup: {selector}")
                    break
            except:
                continue
        
        # Strategy 4: Click overlay backgrounds (many popups close when you click outside)
        overlay_selectors = [
            '[class*="backdrop"]',
            '[class*="overlay"]:not(button)',
            '[class*="modal-backdrop"]',
            '[class*="popup-backdrop"]',
            '[style*="position: fixed"][style*="z-index"]',
        ]
        
        for selector in overlay_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    # Check if it's actually a backdrop (large area)
                    box = await element.bounding_box()
                    if box and box['width'] > 500 and box['height'] > 500:
                        # Click near the edge of the overlay to avoid content
                        await page.mouse.click(box['x'] + 50, box['y'] + 50)
                        await self._human_like_delay(0.3, 0.8)
                        dismissed_count += 1
                        logger.debug(f"Dismissed overlay: {selector}")
                        break
            except:
                continue
        
        # Strategy 5: Look for popups that might appear after initial load
        await self._human_like_delay(1.0, 2.0)
        
        # Try ESC one more time after delay
        try:
            await page.keyboard.press('Escape')
            await self._human_like_delay(0.5, 1.0)
        except:
            pass
        
        if dismissed_count > 0:
            logger.debug(f"Dismissed {dismissed_count} popup(s)")
    
    async def _detect_remaining_popups(self, page: Page) -> bool:
        """Detect if there are still popups or overlays present on the page."""
        try:
            # Look for common popup/modal indicators
            popup_indicators = [
                '[class*="modal"]:not([style*="display: none"])',
                '[class*="popup"]:not([style*="display: none"])',
                '[class*="overlay"]:not([style*="display: none"])',
                '[class*="dialog"]:not([style*="display: none"])',
                '[role="dialog"]',
                '[role="alertdialog"]',
                '[style*="z-index"]:not(header):not(nav):not(footer)',
            ]
            
            for selector in popup_indicators:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        if await element.is_visible():
                            # Check if it's covering a significant portion of the screen
                            box = await element.bounding_box()
                            if box and (box['width'] > 300 or box['height'] > 200):
                                logger.debug(f"Popup detected: {selector}")
                                return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            return False  # Error detecting popups (non-critical)
    
    async def _capture_product_area_screenshot(self, page: Page, screenshot_path: Path) -> None:
        """Capture a 1920x1080 screenshot from the top of the page with optimized file size."""
        try:
            # Use JPEG with high quality for smaller file size (50-80% reduction vs PNG)
            jpeg_path = screenshot_path.with_suffix('.jpg')
            
            # Capture exactly 1920x1080 from the top-left corner of the page
            await page.screenshot(
                path=jpeg_path,
                type='jpeg',
                quality=90,  # High quality but compressed
                clip={
                    'x': 0,
                    'y': 0,
                    'width': 1920,
                    'height': 1080
                }
            )
            logger.info("📸 Screenshot captured")
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            # Fallback to full page if clip fails
            jpeg_path = screenshot_path.with_suffix('.jpg')
            await page.screenshot(path=jpeg_path, type='jpeg', quality=90, full_page=True)
 
    async def _detect_and_handle_cookie_consent(self, page: Page, url: str) -> bool:
        """Detect and automatically accept cookie consent banners."""
        try:
            # Look for common cookie consent patterns
            cookie_selectors = [
                'button:has-text("Accept all cookies")',
                'button:has-text("Accept All")',
                'button:has-text("Accept")',
                'button:has-text("I agree")',
                '[data-testid*="accept"]',
                '.cookie-accept',
                '#cookie-accept',
            ]
            
            for selector in cookie_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        logger.debug(f"Clicking cookie consent button: {selector}")
                        await element.click()
                        await self._human_like_delay(1.0, 2.0)
                        return True
                except:
                    continue
                    
            return True  # No cookie consent detected or couldn't dismiss
            
        except Exception as e:
            logger.debug(f"Error handling cookie consent: {e}")
            return True

    async def capture_product_page(self, url: str, item_number: int) -> Dict[str, str]:
        """Capture screenshot with fresh Oxylabs connection - stealth delays kept for anti-detection."""
        if not self.browser:
            raise RuntimeError("Browser not initialized")
        
        context = None
        page = None
        try:
            logger.info(f"Capturing product page with fresh Oxylabs IP for item {item_number}: {url}")
            
            # Create completely fresh stealth context - this gets new Oxylabs IP
            logger.debug(f"Item {item_number}: Creating stealth context...")
            context = await asyncio.wait_for(self._create_stealth_context(url), timeout=30.0)
            
            # Create new page
            logger.debug(f"Item {item_number}: Creating new page...")
            page = await context.new_page()
            
            # Navigate to the page with efficient loading
            logger.debug(f"Item {item_number}: Navigating to page...")
            await asyncio.wait_for(
                page.goto(url, wait_until="domcontentloaded", timeout=SCREENSHOT_TIMEOUT),
                timeout=SCREENSHOT_TIMEOUT / 1000.0
            )
            
            # Wait for page to be loaded efficiently
            logger.debug(f"Item {item_number}: Waiting for page to load...")
            await asyncio.wait_for(self._wait_for_page_fully_loaded(page), timeout=30.0)
            
            # Single efficient popup and consent handling
            logger.debug(f"Item {item_number}: Handling overlays...")
            await asyncio.wait_for(self._handle_page_overlays(page), timeout=15.0)
            
            # Move mouse outside viewport to avoid hover effects on product images
            logger.debug(f"Item {item_number}: Moving mouse...")
            await page.mouse.move(-10, -10)  # Move mouse outside the viewport
            await self._human_like_delay(0.2, 0.5)  # Brief delay to ensure hover effects clear
            
            # Take screenshot
            logger.debug(f"Item {item_number}: Taking screenshot...")
            screenshot_path = self.screenshots_dir / f"item_{item_number}.png"
            await asyncio.wait_for(self._capture_product_area_screenshot(page, screenshot_path), timeout=30.0)
            
            # Do OCR fast using thread pool (non-blocking)
            jpeg_path = screenshot_path.with_suffix('.jpg')
            
            # Run OCR in thread pool to avoid blocking event loop, but wait for result
            logger.debug(f"Item {item_number}: Starting OCR...")
            loop = asyncio.get_event_loop()
            ocr_text = await asyncio.wait_for(
                loop.run_in_executor(
                    self.ocr_executor, 
                    self._extract_ocr_fast, 
                    str(jpeg_path), 
                    item_number
                ),
                timeout=60.0  # OCR timeout
            )
            
            product_data = {
                'screenshot_path': str(jpeg_path),
                'ocr_text': ocr_text
            }
            
            logger.info(f"Successfully captured data with fresh IP for item {item_number}")
            return product_data
            
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout capturing product page for item {item_number} at {url}: Operation timed out")
            return {
                'screenshot_path': '',
                'ocr_text': '',
                'error': 'Screenshot operation timed out'
            }
        except Exception as e:
            logger.error(f"Error capturing product page for item {item_number} at {url}: {e}")
            return {
                'screenshot_path': '',
                'ocr_text': '',
                'error': str(e)
            }
        finally:
            # Clean up browser resources even if there was a timeout or error
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup for item {item_number}: {cleanup_error}")

    async def capture_product_page_fresh_connection(self, url: str, item_number: int) -> Dict[str, str]:
        """Alias for capture_product_page - already creates fresh connections."""
        return await self.capture_product_page(url, item_number)
    
    async def process_urls_by_domain(self, url_mapping: Dict[int, str]) -> Dict[int, Dict[str, str]]:
        """Process URLs using domain-based async queues for better stealth."""
        results = {}
        
        # Group URLs by domain
        domain_groups = defaultdict(list)
        for item_number, url in url_mapping.items():
            if url and url != "NOT_FOUND":
                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower()
                    # Remove www. prefix for grouping
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    domain_groups[domain].append((item_number, url))
                except Exception as e:
                    logger.error(f"Error parsing URL {url}: {e}")
                    results[item_number] = {
                        'screenshot_path': '',
                        'ocr_text': '',
                        'error': f'Invalid URL: {str(e)}'
                    }
        
        if not domain_groups:
            logger.warning("No valid URLs to process for screenshots")
            return results
        
        logger.info(f"Processing {len(url_mapping)} URLs across {len(domain_groups)} domains")
        for domain, urls in domain_groups.items():
            logger.info(f"  {domain}: {len(urls)} URLs")
        
        # Create async tasks for each domain
        domain_tasks = []
        for domain, urls in domain_groups.items():
            task = asyncio.create_task(
                self._process_domain_queue(domain, urls), 
                name=f"domain-{domain}"
            )
            domain_tasks.append(task)
        
        # Wait for all domain tasks to complete
        domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)
        
        # Combine results from all domains
        for i, result in enumerate(domain_results):
            if isinstance(result, Exception):
                domain = list(domain_groups.keys())[i]
                logger.error(f"Domain task failed for {domain}: {result}")
                # Mark all URLs from this domain as failed
                for item_number, url in domain_groups[domain]:
                    results[item_number] = {
                        'screenshot_path': '',
                        'ocr_text': '',
                        'error': f'Domain task failed: {str(result)}'
                    }
            else:
                results.update(result)
        
        return results
    
    async def _process_domain_queue(self, domain: str, urls: List[Tuple[int, str]]) -> Dict[int, Dict[str, str]]:
        """Process all URLs for a specific domain - NO DELAYS with Oxylabs proxy rotation."""
        results = {}
        
        logger.info(f"Processing {domain} with {len(urls)} URLs - NO DELAYS, fresh Oxylabs IP per request")
        
        # Process in parallel - each creates fresh connection for IP rotation
        async def process_single_url(item_number: int, url: str) -> tuple[int, Dict[str, str]]:
            try:
                logger.info(f"Processing item {item_number} with fresh Oxylabs connection")
                data = await self.capture_product_page(url, item_number)
                return item_number, data
            except Exception as e:
                logger.error(f"Failed to process URL for item {item_number}: {e}")
                return item_number, {
                    'screenshot_path': '',
                    'ocr_text': '',
                    'error': str(e)
                }
        
        # Process all URLs in parallel - each gets fresh Oxylabs IP
        tasks = [process_single_url(item_number, url) for item_number, url in urls]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
            else:
                item_number, data = result
                results[item_number] = data
        
        logger.info(f"Completed processing {domain}: {len(results)} results with fresh IPs")
        return results

    async def process_urls(self, url_mapping: Dict[int, str]) -> Dict[int, Dict[str, str]]:
        """Process multiple URLs using domain-based queues (default method)."""
        return await self.process_urls_by_domain(url_mapping)
    
    async def process_urls_sequential(self, url_mapping: Dict[int, str]) -> Dict[int, Dict[str, str]]:
        """Process multiple URLs sequentially (fallback method)."""
        results = {}
        
        # Randomize order to appear more human-like
        items = list(url_mapping.items())
        random.shuffle(items)
        
        for item_number, url in items:
            if url and url != "NOT_FOUND":
                try:
                    data = await self.capture_product_page(url, item_number)
                    results[item_number] = data
                    
                    # Add delay between requests to avoid rate limiting
                    await self._human_like_delay(5.0, 10.0)
                    
                except Exception as e:
                    logger.error(f"Failed to process URL for item {item_number}: {e}")
                    results[item_number] = {
                        'screenshot_path': '',
                        'ocr_text': '',
                        'error': str(e)
                    }
        
        return results