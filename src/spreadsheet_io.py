import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SpreadsheetHandler:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_spreadsheet(self) -> pd.DataFrame:
        """Load the Excel spreadsheet and parse the data."""
        try:
            # Load the Excel file
            self.df = pd.read_excel(self.file_path, engine='openpyxl')
            logger.info(f"Loaded spreadsheet with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Based on the examination, the actual data starts at row 7 (index 7)
            # Row 6 contains the headers: Item #, Room, Brand or Manufacturer, etc.
            # and the columns appear to be:
            # Col 0: Item number
            # Col 1: Room/Category (Kitchen, etc.)
            # Col 2: Brand/Make
            # Col 3: Model# (often unused/NaN)
            # Col 4: Item Description
            # Col 5: Original Vendor
            # Cols 6-11: Various pricing and quantity data
            # Col 8: Item Age (Months)
            
            # Filter to actual product rows (skip header rows)
            product_rows = []
            for idx, row in self.df.iterrows():
                # Skip rows that don't have proper data structure
                # Data starts at row 7 (index 7), row 6 has headers
                if (idx < 7 or 
                    pd.isna(row.iloc[0]) or 
                    not isinstance(row.iloc[0], (int, float)) or
                    pd.isna(row.iloc[4])):  # Description column
                    continue
                    
                # Extract the relevant data
                item_data = {
                    'item_number': int(row.iloc[0]),
                    'room': str(row.iloc[1]) if not pd.isna(row.iloc[1]) else '',
                    'make': str(row.iloc[2]) if not pd.isna(row.iloc[2]) else '',
                    'model': '',  # Col 3 seems unused
                    'description': str(row.iloc[4]),
                    'original_vendor': str(row.iloc[5]) if not pd.isna(row.iloc[5]) else '',
                    'quantity_lost': int(row.iloc[6]) if not pd.isna(row.iloc[6]) and isinstance(row.iloc[6], (int, float)) else 1,
                    'item_age_months': int(row.iloc[8]) if not pd.isna(row.iloc[8]) and isinstance(row.iloc[8], (int, float)) else None,
                    'condition': str(row.iloc[9]) if not pd.isna(row.iloc[9]) else '',
                    'cost_to_replace_each': float(row.iloc[10]) if not pd.isna(row.iloc[10]) and isinstance(row.iloc[10], (int, float)) else 0.0,
                    'total_cost': float(row.iloc[11]) if not pd.isna(row.iloc[11]) and isinstance(row.iloc[11], (int, float)) else 0.0,
                    'price': self._extract_price(row),
                    'original_row_index': idx
                }
                product_rows.append(item_data)
            
            # Create a clean DataFrame
            self.df = pd.DataFrame(product_rows)
            logger.info(f"Parsed {len(self.df)} product rows")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading spreadsheet: {e}")
            raise
    
    def _extract_price(self, row) -> float:
        """Extract price from the row data."""
        # Price seems to be in the last columns (10, 11)
        price_candidates = [row.iloc[10], row.iloc[11]]
        
        for candidate in price_candidates:
            if pd.notna(candidate) and isinstance(candidate, (int, float)) and candidate > 0:
                return float(candidate)
        
        # Fallback - look for any numeric value that could be a price
        for i in range(6, 12):
            val = row.iloc[i]
            if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                return float(val)
        
        return 0.0
    
    def add_url_column(self, urls: Dict[int, str]):
        """Add a column with matched URLs."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        self.df['matched_url'] = self.df['item_number'].map(urls)
        logger.info(f"Added URLs for {len([u for u in urls.values() if u])} items")
    
    def add_llm_prices(self, prices: Dict[int, str]):
        """Add a column with LLM-extracted prices."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        self.df['llm_extracted_price'] = self.df['item_number'].map(prices)
        logger.info(f"Added LLM-extracted prices for {len([p for p in prices.values() if p])} items")
    
    def add_screenshot_data(self, screenshot_data: Dict[int, Dict]):
        """Add columns with screenshot and extracted data."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        # Add new columns (removed image_url)
        self.df['screenshot_path'] = ''
        self.df['fallback_extracted_price'] = ''
        self.df['extracted_title'] = ''
        
        for item_num, data in screenshot_data.items():
            mask = self.df['item_number'] == item_num
            if mask.any():
                self.df.loc[mask, 'screenshot_path'] = data.get('screenshot_path', '')
                self.df.loc[mask, 'fallback_extracted_price'] = data.get('price', '')
                self.df.loc[mask, 'extracted_title'] = data.get('title', '')
        
        logger.info(f"Added screenshot data for {len(screenshot_data)} items")
    
    def save_spreadsheet(self, output_path: str):
        """Save the updated spreadsheet."""
        if self.df is None:
            raise ValueError("No data to save")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Saved updated spreadsheet to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving spreadsheet: {e}")
            raise
    
    def get_product_items(self) -> List[Dict]:
        """Get list of product items for processing."""
        if self.df is None:
            raise ValueError("No data loaded")
        
        return self.df.to_dict('records') 