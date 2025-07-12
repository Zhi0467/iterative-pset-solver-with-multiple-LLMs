import os
import fitz  # PyMuPDF
import pdfplumber
import pymupdf4llm
from typing import Optional, List, Dict, Any
from PIL import Image
import io
import base64

class AdvancedPDFExtractor:
    """Advanced PDF text extraction with multiple fallback methods and caching."""
    
    def __init__(self):
        self._extractors = [
            self._extract_with_pymupdf4llm,
            self._extract_with_pdfplumber,
            self._extract_with_pymupdf_basic
        ]
        self._text_cache = {}  # Cache for extracted text
        self._max_cache_size = 50  # Maximum number of PDFs to cache
        
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with caching and quality assessment."""
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} was not found.")
            
        # Check cache first
        if pdf_path in self._text_cache:
            print(f"📋 Using cached text for {pdf_path}")
            return self._text_cache[pdf_path]
            
        #print(f"📄 Extracting text from {pdf_path}...")
        
        # Try each extractor and assess quality
        results = []
        last_error = None
        
        for extractor in self._extractors:
            try:
                text = extractor(pdf_path)
                if text and text.strip():
                    results.append({
                        'method': extractor.__name__,
                        'text': text,
                        'quality_score': self._assess_quality(text)
                    })
            except Exception as e:
                last_error = e
                print(f"Warning: {extractor.__name__} failed: {e}")
                continue
                
        if not results:
            if last_error:
                raise last_error
            raise ValueError(f"No text could be extracted from {pdf_path}. The PDF might be image-based or corrupted.")
            
        # Get the best quality result
        best_result = max(results, key=lambda x: x['quality_score'])
        print(f"Using extraction method: {best_result['method']} (quality score: {best_result['quality_score']:.2f})")
        
        # Cache the best result
        if len(self._text_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._text_cache))
            del self._text_cache[oldest_key]
            
        self._text_cache[pdf_path] = best_result['text']
        print(f"✅ Text cached for {pdf_path}")
        
        return best_result['text']
        
    def clear_cache(self):
        """Clear the text extraction cache."""
        self._text_cache.clear()
        print("🗑️ Cleared PDF text cache")
    
    def _extract_with_pymupdf4llm(self, pdf_path: str) -> str:
        """Extract using pymupdf4llm - best for LLM processing."""
        try:
            # This method is specifically designed for LLM consumption
            # It handles mathematical formulas, tables, and maintains structure
            text = pymupdf4llm.to_markdown(pdf_path)
            return text
        except Exception as e:
            raise Exception(f"pymupdf4llm extraction failed: {e}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract using pdfplumber - good for tables and structured content."""
        try:
            text_parts = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = f"\\n--- Page {page_num + 1} ---\\n"
                    
                    # Extract regular text
                    if page.extract_text():
                        page_text += page.extract_text() + "\\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        page_text += f"\\n**Table {i+1}:**\\n"
                        for row in table:
                            if row:  # Skip empty rows
                                page_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\\n"
                    
                    text_parts.append(page_text)
            
            return "\\n".join(text_parts)
        except Exception as e:
            raise Exception(f"pdfplumber extraction failed: {e}")
    
    def _extract_with_pymupdf_basic(self, pdf_path: str) -> str:
        """Basic PyMuPDF extraction - fallback method."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = f"\\n--- Page {page_num + 1} ---\\n"
                
                # Try to extract text
                text = page.get_text()
                if text.strip():
                    page_text += text + "\\n"
                
                # If no text found, try OCR on images (basic approach)
                if not text.strip():
                    # Get page as image and attempt basic description
                    page_text += "[This page appears to contain images or non-text content]\\n"
                
                text_parts.append(page_text)
            
            doc.close()
            return "\\n".join(text_parts)
        except Exception as e:
            raise Exception(f"PyMuPDF basic extraction failed: {e}")
    
    def _assess_quality(self, text: str) -> float:
        """
        Assess the quality of extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score (0-1, higher is better)
        """
        if not text or not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor (longer text generally better, but with diminishing returns)
        length_score = min(len(text) / 5000, 1.0) * 0.3
        score += length_score
        
        # Mathematical content indicators
        math_indicators = ['=', '+', '-', '×', '÷', '∫', '∑', '√', 'matrix', 'equation']
        math_score = sum(1 for indicator in math_indicators if indicator.lower() in text.lower()) / len(math_indicators) * 0.3
        score += math_score
        
        # Structure indicators
        structure_indicators = ['Exercise', 'Problem', 'Solution', 'Chapter', 'Section']
        structure_score = sum(1 for indicator in structure_indicators if indicator in text) / len(structure_indicators) * 0.2
        score += structure_score
        
        # Character diversity (avoid repetitive extraction errors)
        unique_chars = len(set(text.lower()))
        diversity_score = min(unique_chars / 50, 1.0) * 0.2
        score += diversity_score
        
        return min(score, 1.0)
    
    def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF for potential OCR processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of image information dictionaries
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_info = {
                                'page': page_num + 1,
                                'index': img_index,
                                'data': img_data,
                                'size': (pix.width, pix.height)
                            }
                            images.append(img_info)
                        
                        pix = None
                    except Exception as e:
                        print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Image extraction failed: {e}")
        
        return images 