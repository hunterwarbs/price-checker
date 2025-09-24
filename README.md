# AI Product Price Checker

An intelligent system that reads product information from Excel spreadsheets and automatically finds matching retail product URLs using AI, then captures screenshots and extracts product data.

**Deployment note:** Docker + Cloudflare tunnel setup is now available. See `DOCKER_README.md` for compose profiles, tunnel instructions, and environment variables.

## Features

- **AI-Powered Search**: Uses OpenRouter LLMs with Oxylabs search tooling to find the best matching retail product pages
- **Intelligent Verification**: LLM analyzes search results and page content to confirm correct product matches
- **Automated Screenshots**: Uses Playwright to capture full-page screenshots of product pages
- **Data Extraction**: Uses Fireworks AI Qwen2.5-VL for OCR on screenshots to extract text (prices, titles)
- **Excel Integration**: Reads from and writes to Excel files with all collected data

## Architecture

The system consists of several key components:

1. **Spreadsheet I/O** (`src/spreadsheet_io.py`) - Handles reading and writing Excel files
2. **LLM Search Agent** (`src/llm_search.py`) - Integrates OpenRouter and Oxylabs search for intelligent product search
3. **Screenshot Capture** (`src/screenshotter.py`) - Uses Playwright for web page capture and data extraction
4. **OCR** (`src/ocr_analyzer.py`) - Fireworks AI Qwen2.5-VL-based OCR service
5. **Main CLI** (`src/main.py`) - Command-line interface for the entire system

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright Browser

```bash
python -m playwright install chromium
```

### 3. Set Up API Keys

Copy the example environment file and add your secrets:

```bash
cp env.example .env
```

Edit `.env` and add the required values:

```
OPENROUTER_API_KEY=sk-or-your-actual-api-key-here
FIREWORKS_API_KEY=your-fireworks-api-key
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_VL_MODEL=accounts/fireworks/models/qwen2p5-vl-32b-instruct
OXYLABS_WEB_API=your-web-api-username:your-web-api-password
OXYLABS_PROXY=your-proxy-username:your-proxy-password
```

If you plan to expose the app via the Cloudflare tunnel compose profile, also set:

```
CLOUDFLARE_TUNNEL_TOKEN=your-tunnel-token-here
```

**Getting API Keys:**

- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key
- **Fireworks AI**: Sign up at [fireworks.ai](https://fireworks.ai/) and create an API key
- **Oxylabs Web API**: Sign up for the Oxylabs SERP API and create credentials (username/password format)
- **Oxylabs Proxy**: Generate proxy credentials in your Oxylabs dashboard

## Usage

### Basic Usage

Process the default Excel file:

```bash
python -m src.main
```

### Custom Input/Output

```bash
python -m src.main --input path/to/your/file.xlsx --output results.xlsx
```

### Test with Limited Items

Process only the first 5 items (useful for testing):

```bash
python -m src.main --limit 5
```

### Verbose Logging

```bash
python -m src.main --verbose
```

### All Options

```bash
python -m src.main --input data/custom.xlsx --output results.xlsx --limit 10 --verbose
```

## Input File Format

The system expects an Excel file with the following structure (based on the Xact Contents Import format):

- **Column 0**: Item number
- **Column 1**: Category/Room (e.g., "Kitchen")
- **Column 2**: Brand/Make
- **Column 4**: Product description
- **Column 5**: Store/Source
- **Columns 10-11**: Price information

The system automatically detects and skips header rows, processing only rows with valid product data.

## Output

The system generates:

1. **Updated Excel file** with additional columns:
   - `matched_url`: The found retail product URL
   - `screenshot_path`: Path to the captured screenshot
   - `extracted_price`: Price found on the product page
   - `extracted_title`: Product title from the page
   - `extracted_image_url`: Main product image URL

2. **Screenshots folder** (`screenshots/`) containing PNG files for each product page

3. **Log file** (`price_checker.log`) with detailed processing information

## How It Works

### 1. Spreadsheet Parsing
- Loads Excel file and identifies product data rows
- Extracts make, model, description, and price for each item
- Skips header rows and invalid entries automatically

### 2. AI-Powered URL Discovery
- For each product, the LLM agent:
  - Uses Oxylabs Web API to search for matching products
  - Analyzes search results and page content
  - Verifies the match quality
  - Returns the best matching retail URL

### 3. Screenshot and Data Extraction
- Uses Playwright to visit each found URL
- Captures top-of-page screenshots
- Sends screenshots to Fireworks Qwen2.5-VL OCR to extract visible text and pricing

### 4. Results Compilation
- Combines all collected data into the output Excel file
- Provides detailed logging and progress reporting

## Configuration

Key settings can be modified in `src/config.py`:

- **Model Selection**: Change the OpenRouter model used
- **OCR Model**: Configure Fireworks base URL and VLM model
- **Screenshot Settings**: Modify viewport size and timeout values
- **File Paths**: Customize input/output locations

## Error Handling

The system includes robust error handling:

- **API Failures**: Automatic retries with exponential backoff
- **Network Issues**: Graceful handling of timeouts and connection errors
- **Invalid URLs**: Continues processing even if some URLs fail
- **Data Extraction**: Vision OCR via Fireworks with basic normalization

## Logging

Comprehensive logging is provided:

- **Console Output**: Real-time progress and status updates
- **Log File**: Detailed information saved to `price_checker.log`
- **Verbose Mode**: Additional debug information when needed

## Performance Considerations

- **Rate Limiting**: Built-in delays to respect API rate limits
- **Parallel Processing**: Screenshots are captured efficiently
- **Memory Management**: Processes items in batches to manage memory usage
- **Cost Control**: Use `--limit` option to test with fewer items first

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file is properly configured with valid API keys
2. **Playwright Issues**: Run `python -m playwright install chromium` if browser installation fails
3. **Excel File Errors**: Verify the input file format matches expected structure
4. **Network Timeouts**: Some websites may be slow; the system will continue with other items

### Getting Help

- Check the log file (`price_checker.log`) for detailed error information
- Use `--verbose` flag for additional debugging output
- Ensure all dependencies are properly installed

## Example Output

```
✅ Processing complete! Results saved to: data/with_urls_and_screenshots.xlsx

=== PROCESSING COMPLETE ===
Total items processed: 25
URLs found: 23
Screenshots captured: 21
Results saved to: data/with_urls_and_screenshots.xlsx
Screenshots saved to: screenshots/
```

## License

This project is provided as-is for demonstration purposes. 