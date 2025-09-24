#!/usr/bin/env python3
"""
AI-powered product price checker and URL matcher.

This tool reads an Excel spreadsheet with product information and:
1. Uses LLM + Exa AI to find matching retail product URLs
2. Takes screenshots of product pages using Playwright
3. Extracts product data (price, title, image) from the pages
4. Outputs an updated Excel file with all the collected data
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import INPUT_FILE, OUTPUT_FILE
from .spreadsheet_io import SpreadsheetHandler
from .matcher import ProductMatcher
from .colored_logging import setup_colored_logging

def setup_logging(verbose: bool = False):
    """Set up colored logging configuration."""
    setup_colored_logging(verbose)

def validate_environment():
    """Validate that required environment variables are set."""
    from .config import OPENROUTER_API_KEY, OXYLABS_WEB_API_USERNAME, OXYLABS_WEB_API_PASSWORD
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it in your .env file or environment")
        return False
    
    if not OXYLABS_WEB_API_USERNAME or not OXYLABS_WEB_API_PASSWORD:
        print("ERROR: OXYLABS_WEB_API environment variable is not set or invalid")
        print("Please set it in your .env file in format 'username:password'")
        return False
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-powered product price checker and URL matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                                    # Process default file
  python -m src.main --input custom.xlsx               # Process custom file
  python -m src.main --limit 10                        # Process only first 10 items
  python -m src.main --output results.xlsx --verbose   # Custom output with verbose logging
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=INPUT_FILE,
        help=f'Input Excel file path (default: {INPUT_FILE})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output Excel file path (default: {OUTPUT_FILE})'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit processing to first N items (useful for testing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    try:
        logger.info("🚀 Starting price checker")
        logger.info(f"📄 Input: {Path(input_path).name}")
        if args.limit:
            logger.info(f"🔢 Processing {args.limit} items")
        
        # Step 1: Load spreadsheet  
        logger.info("📊 Loading spreadsheet...")
        spreadsheet = SpreadsheetHandler(str(input_path))
        df = spreadsheet.load_spreadsheet()
        items = spreadsheet.get_product_items()
        
        logger.info(f"✅ Loaded {len(items)} products")
        
        # Step 2: Process items (find URLs and capture screenshots) with limited concurrency
        logger.info("🔍 Finding products and prices...")
        matcher = ProductMatcher()
        url_mapping, llm_price_mapping, screenshot_data = matcher.process_items(items, args.limit)
        
        # Step 3: Update spreadsheet with results
        logger.info("📝 Updating spreadsheet...")
        spreadsheet.add_url_column(url_mapping)
        spreadsheet.add_llm_prices(llm_price_mapping)
        spreadsheet.add_screenshot_data(screenshot_data)
        
        # Step 4: Save results
        logger.info("💾 Saving results...")
        spreadsheet.save_spreadsheet(args.output)
        
        # Summary
        total_items = len(items) if not args.limit else min(len(items), args.limit)
        urls_found = len([url for url in url_mapping.values() if url and url not in ["NOT_FOUND", "ERROR"]])
        llm_prices_found = len([price for price in llm_price_mapping.values() if price])
        screenshots_taken = len([data for data in screenshot_data.values() if data.get('screenshot_path')])
        
        logger.info("🎉 Processing complete!")
        logger.info(f"✅ {urls_found}/{total_items} products found ({llm_prices_found} with prices)")
        logger.info(f"📸 {screenshots_taken} screenshots captured")
        logger.info(f"💾 Results saved to: {Path(args.output).name}")
        
        print(f"\n✅ Processing complete! Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 