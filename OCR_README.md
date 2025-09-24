# OCR Integration for Price Checker

## Overview

The price checker now includes OCR (Optical Character Recognition) functionality that extracts text from screenshots and provides it to the LLM for analysis. This allows the LLM to:

1. **Verify screenshot quality** - Determine if the page loaded correctly or shows errors
2. **Extract pricing information** - Get prices that may not be available in HTML content  
3. **Make intelligent retry decisions** - Try another URL if the OCR shows "not available" or error pages
4. **Improve accuracy** - Use visual text extraction instead of just HTML parsing

## How It Works

### 1. Screenshot + OCR Flow
```
1. LLM searches for product URLs using Exa
2. LLM calls take_screenshot tool with a URL
3. Screenshot is captured using Playwright
4. OCR extracts text from the screenshot image
5. Raw OCR text is returned to the LLM
6. LLM analyzes OCR text to extract price and validate page
```

### 2. LLM Decision Making
The LLM now receives OCR text and can:
- **Extract prices** directly from the visual text
- **Detect errors** like "404 Not Found", "Out of Stock", "Item Unavailable"
- **Validate products** by reading product titles and descriptions
- **Retry with different URLs** if OCR shows the page is invalid

### 3. Integration Points

#### OCR Analyzer (`src/ocr_analyzer.py`)
- Simple text extraction using Tesseract OCR
- Image preprocessing for better OCR accuracy
- Text cleaning and normalization

#### LLM Search Agent (`src/llm_search.py`)  
- Updated `take_screenshot` tool to include OCR text in response
- LLM receives OCR text and makes decisions based on it

#### System Prompts (`src/config.py`)
- Updated to explain OCR functionality to the LLM
- Instructions on how to analyze OCR text for price extraction
- Guidance on when to retry with different URLs

## Installation Requirements

The system now requires additional OCR dependencies:

```bash
# Install Tesseract OCR system package
sudo apt-get install tesseract-ocr tesseract-ocr-eng  # Ubuntu/Debian
# or
brew install tesseract  # macOS

# Python packages (already in requirements.txt)
pip install opencv-python pytesseract Pillow
```

## Benefits

### 1. Better Price Extraction
- Captures dynamically loaded prices that don't appear in HTML
- Handles JavaScript-rendered pricing
- Works with images/graphics containing price text

### 2. Intelligent Error Handling  
- LLM can detect when a page shows "Out of Stock" or errors
- Automatically tries alternative URLs instead of returning no results
- Reduces false negatives from temporary page issues

### 3. Visual Validation
- Confirms the screenshot actually shows the intended product
- Detects when search results lead to wrong products
- Validates that pages loaded correctly

## Example OCR Text Analysis

When the LLM receives OCR text like:
```
"MacBook Pro 16-inch M3 Pro
$2,499.00
Add to Cart
Free Shipping
In Stock"
```

The LLM can:
1. Confirm this is the right product (MacBook Pro 16-inch M3 Pro)
2. Extract the price ($2,499.00)  
3. Verify availability (In Stock)
4. Return both URL and price with confidence

If OCR text shows:
```
"404 Error
Page Not Found  
The requested item is no longer available"
```

The LLM will:
1. Recognize this as an invalid page
2. Try the next most promising URL from search results
3. Continue until finding a valid product page

## Configuration

The OCR system uses conservative settings optimized for web page text:
- LSTM OCR Engine (--oem 3)
- Uniform text block processing (--psm 6)
- Image preprocessing (grayscale, denoising, contrast enhancement)

No additional configuration needed - the system works automatically. 