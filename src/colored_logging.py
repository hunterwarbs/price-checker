import logging
import re
from typing import Dict, Optional

class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that colors log messages based on component and level.
    """
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        
        # Component colors
        'llm': '\033[94m',       # Blue
        'exa': '\033[92m',       # Green  
        'playwright': '\033[95m', # Magenta
        'main': '\033[96m',      # Cyan
        'matcher': '\033[33m',   # Orange/Yellow
        'spreadsheet': '\033[35m', # Purple
        'default': '\033[37m',   # White
        
        # Level colors
        'error': '\033[91m',     # Red
        'warning': '\033[93m',   # Yellow
        'info': '\033[37m',      # White
        'debug': '\033[90m',     # Dark Gray
    }
    
    # Component patterns to identify log source
    COMPONENT_PATTERNS = {
        'llm': [
            r'llm_search',
            r'openai',
            r'openrouter',
            r'🔎.*Searching for:',
            r'🔍.*Getting page content',
            r'✅.*Successfully retrieved content',
            r'Executing.*with args',
            r'Found URL.*for item',
            r'No valid URL found',
            r'Error in find_product_url',
            r'Searching for item \d+:',
        ],
        'exa': [
            r'📊.*Found.*search results',
            r'⚠️.*No content returned',
            r'❌.*Error in web search',
            r'❌.*Error getting page content',
            r'Finding similar pages to:',
            r'Error finding similar pages',
        ],
        'playwright': [
            r'screenshotter',
            r'Capturing product page',
            r'Successfully captured data',
            r'Captured.*screenshot',
            r'Error capturing screenshot',
            r'Navigation attempt.*failed',
            r'All navigation attempts failed',
            r'Processing.*\[.*\]:',
            r'Starting domain queue',
            r'Completed domain queue',
            r'Failed to process URL',
            r'Domain task failed',
            r'DOM content failed to load',
            r'No content elements found',
        ],
        'main': [
            r'__main__',
            r'Starting AI Product Price Checker',
            r'Step \d+:',
            r'=== PROCESSING COMPLETE ===',
            r'Total items processed:',
            r'URLs found:',
            r'Screenshots captured:',
            r'Results saved to:',
            r'Process interrupted',
        ],
        'matcher': [
            r'matcher',
            r'Starting.*matching pipeline',
            r'Phase \d+:',
            r'parallel LLM search tasks',
            r'Async.*search complete',
            r'Screenshot capture complete',
            r'Finding URLs and prices for.*items',
            r'✓ Found URL for item',
            r'✗ No URL found for item',
            r'Started domain processor',
            r'Domain processor.*received stop signal',
            r'Processing.*: item',
            r'✓ Completed item.*from',
            r'Failed to process item.*from',
        ],
        'spreadsheet': [
            r'spreadsheet_io',
            r'Loaded spreadsheet with',
            r'Parsed.*product rows',
            r'Added URLs for.*items',
            r'Added.*prices for.*items',
            r'Added screenshot data',
            r'Saved updated spreadsheet',
        ]
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        
    def _get_component_color(self, record: logging.LogRecord) -> str:
        """Determine the appropriate color based on the log record."""
        # First check if it's an error (highest priority)
        if record.levelno >= logging.ERROR:
            return self.COLORS['error']
        
        # Check logger name and message content for component identification
        log_content = f"{record.name} {record.getMessage()}"
        
        # Check each component pattern
        for component, patterns in self.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    return self.COLORS[component]
        
        # Check for warning level
        if record.levelno >= logging.WARNING:
            return self.COLORS['warning']
            
        # Default color based on level
        level_colors = {
            logging.DEBUG: self.COLORS['debug'],
            logging.INFO: self.COLORS['info'],
            logging.WARNING: self.COLORS['warning'],
            logging.ERROR: self.COLORS['error'],
            logging.CRITICAL: self.COLORS['error'],
        }
        
        return level_colors.get(record.levelno, self.COLORS['default'])
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with appropriate colors."""
        # Format the message using the parent formatter
        message = super().format(record)
        
        if not self.use_colors:
            return message
        
        # Get the appropriate color
        color = self._get_component_color(record)
        
        # Apply color to the entire message
        colored_message = f"{color}{message}{self.COLORS['reset']}"
        
        return colored_message


class ColoredStreamHandler(logging.StreamHandler):
    """
    Stream handler that only uses colors when outputting to a terminal.
    """
    
    def __init__(self, stream=None):
        super().__init__(stream)
        # Check if we're writing to a terminal
        self.use_colors = hasattr(self.stream, 'isatty') and self.stream.isatty()
    
    def setFormatter(self, formatter):
        if isinstance(formatter, ColoredFormatter):
            formatter.use_colors = self.use_colors
        super().setFormatter(formatter)


def setup_colored_logging(verbose: bool = False) -> None:
    """
    Set up colored logging configuration.
    
    Args:
        verbose: If True, set DEBUG level, otherwise INFO level
    """
    import sys
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create colored formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler with color support
    console_handler = ColoredStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create file handler without colors
    file_handler = logging.FileHandler('price_checker.log', mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set levels for specific loggers to reduce noise
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
    logging.getLogger('playwright').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING) 