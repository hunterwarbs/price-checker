#!/usr/bin/env python3
"""
Simple OCR text extraction from product screenshots.
"""

import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Try to import EasyOCR for better accuracy and GPU support
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available, falling back to Tesseract. Install with: pip install easyocr")

logger = logging.getLogger(__name__)

class OCRAnalyzer:
    def __init__(self):
        # Initialize EasyOCR if available (much better for web screenshots)
        if EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR with GPU support if available
                self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
                self.use_easyocr = True
                logger.info("EasyOCR initialized with GPU support")
            except Exception as e:
                try:
                    # Fallback to CPU if GPU fails
                    self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    self.use_easyocr = True
                    logger.info("EasyOCR initialized with CPU (GPU failed)")
                except Exception as e2:
                    logger.warning(f"EasyOCR initialization failed: {e2}, using Tesseract")
                    self.use_easyocr = False
        else:
            self.use_easyocr = False
        
        # Tesseract fallback configurations
        self.ocr_configs = [
            '--oem 3 --psm 3',   # Fully automatic page segmentation
            '--oem 3 --psm 4',   # Single column of text of variable sizes  
            '--oem 3 --psm 6',   # Uniform text block (original)
            '--oem 3 --psm 11',  # Sparse text, find as much as possible
            '--oem 3 --psm 12',  # Sparse text with OSD
        ]

    def extract_text_from_image(self, image_path: str, fast_mode: bool = True) -> str:
        """Extract text from an image using OCR with multiple approaches."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return ""
            
            # Use EasyOCR if available (much better for web screenshots)
            if self.use_easyocr:
                return self._extract_with_easyocr(str(image_path))
            
            # Fallback to Tesseract
            logger.debug("Using Tesseract OCR")
            
            # Load image using OpenCV for preprocessing
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return ""
            
            if fast_mode:
                # Fast mode: try 2 best preprocessing approaches with 2 best configs
                best_text = ""
                best_length = 0
                
                # Try the two best preprocessing approaches for web content
                approaches = [
                    ("scaled", self._preprocess_scaled),     # Good for small web text
                    ("adaptive", self._preprocess_adaptive)  # Good for varying backgrounds
                ]
                
                # Try the two most reliable configs
                configs = [
                    '--oem 3 --psm 3',   # Fully automatic page segmentation
                    '--oem 3 --psm 6',   # Uniform text block
                ]
                
                for approach_name, preprocess_func in approaches:
                    processed_img = preprocess_func(img)
                    pil_img = Image.fromarray(processed_img)
                    
                    for config in configs:
                        try:
                            text = pytesseract.image_to_string(pil_img, config=config)
                            cleaned_text = self._clean_extracted_text(text)
                            
                            # Keep the result with the most content (usually more accurate)
                            if len(cleaned_text.strip()) > best_length:
                                best_text = cleaned_text
                                best_length = len(cleaned_text.strip())
                                logger.debug(f"Better result: {approach_name} + {config} ({best_length} chars)")
                                
                        except Exception as e:
                            logger.debug(f"Fast OCR failed with {approach_name} + {config}: {e}")
                            continue
                
                # Fallback if nothing worked
                if not best_text:
                    processed_img = self._preprocess_scaled(img)
                    pil_img = Image.fromarray(processed_img)
                    text = pytesseract.image_to_string(pil_img, config=configs[0])
                    best_text = self._clean_extracted_text(text)
                
                logger.debug(f"Fast OCR extracted {len(best_text)} characters from {image_path.name}")
                return best_text
            
            # Full mode: try all approaches (original logic)
            best_text = ""
            best_confidence = 0
            
            # Try multiple approaches
            approaches = [
                ("standard", self._preprocess_standard),
                ("high_contrast", self._preprocess_high_contrast),
                ("scaled", self._preprocess_scaled),
                ("adaptive", self._preprocess_adaptive)
            ]
            
            for approach_name, preprocess_func in approaches:
                processed_img = preprocess_func(img)
                
                for config in self.ocr_configs:
                    try:
                        # Convert to PIL Image for Tesseract
                        pil_img = Image.fromarray(processed_img)
                        
                        # Extract text using Tesseract
                        text = pytesseract.image_to_string(pil_img, config=config)
                        cleaned_text = self._clean_extracted_text(text)
                        
                        # Get confidence score
                        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Keep the result with highest confidence and reasonable length
                        if avg_confidence > best_confidence and len(cleaned_text.strip()) > 50:
                            best_text = cleaned_text
                            best_confidence = avg_confidence
                            logger.info(f"Best result so far: {approach_name} + {config} (confidence: {avg_confidence:.1f})")
                            
                    except Exception as e:
                        logger.debug(f"Failed with {approach_name} + {config}: {e}")
                        continue
            
            # If no good result, use the standard approach as fallback
            if not best_text:
                processed_img = self._preprocess_standard(img)
                pil_img = Image.fromarray(processed_img)
                text = pytesseract.image_to_string(pil_img, config=self.ocr_configs[0])
                best_text = self._clean_extracted_text(text)
            
            logger.info(f"Final OCR text (confidence: {best_confidence:.1f}): {best_text}...")
            logger.info(f"OCR extracted {len(best_text)} characters from {image_path.name}")
            return best_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def _extract_with_easyocr(self, image_path: str) -> str:
        """Extract text using EasyOCR (GPU-accelerated, better for web content)."""
        try:
            # EasyOCR works directly on image files, no preprocessing needed
            results = self.easyocr_reader.readtext(image_path)
            
            # Extract text from results and join with spaces
            extracted_texts = []
            for (bbox, text, confidence) in results:
                # Only include text with reasonable confidence
                if confidence > 0.3:  # Adjust threshold as needed
                    extracted_texts.append(text)
            
            # Join all text parts with spaces and clean
            full_text = ' '.join(extracted_texts)
            cleaned_text = self._clean_extracted_text(full_text)
            
            logger.debug(f"EasyOCR extracted {len(cleaned_text)} characters with {len(results)} text regions")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def _preprocess_standard(self, img: np.ndarray) -> np.ndarray:
        """Standard preprocessing - original method."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def _preprocess_high_contrast(self, img: np.ndarray) -> np.ndarray:
        """High contrast preprocessing for web text."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Use adaptive threshold for varying lighting
        binary = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _preprocess_scaled(self, img: np.ndarray) -> np.ndarray:
        """Scale image up for better OCR on small text."""
        # Scale image by 2x for better small text recognition
        height, width = img.shape[:2]
        scaled = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising after scaling
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Use Otsu thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _preprocess_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing that works well with web content."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use adaptive threshold which works better with varying backgrounds
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 15, 10)
        
        # Invert if most of the image is dark (common with dark mode websites)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted OCR text."""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        
        # Fix common OCR errors
        text = re.sub(r'[|¦]', 'I', text)  # Pipe symbols often should be I
        text = re.sub(r'[°º]', 'o', text)  # Degree symbols often should be o
        text = re.sub(r'[''`]', "'", text)  # Various quote marks to apostrophe
        text = re.sub(r'["""]', '"', text)  # Various quote marks to straight quotes
        
        # Clean up price patterns
        text = re.sub(r'[\$\s]*(\d+[,\.]\d+)', r'$\1', text)
        
        # Remove obvious OCR artifacts (isolated special characters)
        text = re.sub(r'\s+[^\w\s\$\.\,\-\(\)\%\!\?\:\;\'\"]+\s+', ' ', text)
        
        return text.strip()
    
 