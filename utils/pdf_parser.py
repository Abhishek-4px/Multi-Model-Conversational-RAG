import pymupdf
import os
from typing import List, Dict, Any
import re

class MultimodalPDFParser:
    
    #PDF Parser fn
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        self.images_dir = "extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)
    
    def open(self):
        self.doc = pymupdf.open(self.pdf_path)
        print(f"Opened PDF: {self.pdf_path}")
        print(f"Total pages: {len(self.doc)}")
    
    def close(self):
        if self.doc:
            self.doc.close()

    #Taken string text as input and returns a bool value depending whether maths formulas are found or not
    def detect_mathematical_content(self, text: str) -> bool:
        math_patterns = [
            r'√', 
            r'∠',  
            r'°', 
            r'tan\s*\d+°',
            r'sin\s*\d+°',
            r'cos\s*\d+°',
            r'Δ',  
            r'=\s*\d+',
            r'\d+\s*m', 
            r'Fig\.\s*\d+\.\d+',  
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def extract_text_from_page(self, page_num: int) -> str:
        page = self.doc[page_num]
        text = page.get_text()
        return text
    
    def extract_images_from_page(self, page_num: int) -> List[Dict[str, Any]]:
        page = self.doc[page_num]
        image_list = page.get_images(full=True)
        extracted_images = []
        
        # Only extract meaningful images
        MIN_IMAGE_SIZE = 10000  # Skip images smaller than 10KB
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            try:
                # Extract image
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Skip tiny images
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue
                
                image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
                image_path = os.path.join(self.images_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                img_data = {
                    "page": page_num,
                    "index": img_index,
                    "path": image_path,
                    "filename": image_filename,
                    "extension": image_ext,
                    "xref": xref,
                    "size": len(image_bytes)
                }
                
                extracted_images.append(img_data)
                print(f" Extracted image: {image_filename}")
                
            except Exception as e:
                print(f" Error extracting image {img_index} from page {page_num + 1}: {e}")
        
        return extracted_images
    
    def extract_blocks_from_page(self, page_num: int) -> List[Dict[str, Any]]:
        page = self.doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        extracted_blocks = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span["text"] + " "
                
                has_math = self.detect_mathematical_content(block_text)
                
                extracted_blocks.append({
                    "type": "text",
                    "content": block_text.strip(),
                    "bbox": block["bbox"],
                    "page": page_num,
                    "has_math": has_math
                })
            elif block["type"] == 1:  # Image block
                extracted_blocks.append({
                    "type": "image",
                    "bbox": block["bbox"],
                    "page": page_num,
                    "image_ref": block.get("image", None)
                })
        
        return extracted_blocks
    
    def parse_full_document(self) -> List[Dict[str, Any]]:
        all_chunks = []
        total_images = 0
        total_math_blocks = 0
        
        print(f"\nProcessing {len(self.doc)} pages...")
        
        for page_num in range(len(self.doc)):
            print(f"\n[Page {page_num + 1}/{len(self.doc)}]")

            page_text = self.extract_text_from_page(page_num)

            images = self.extract_images_from_page(page_num)
            total_images += len(images)

            blocks = self.extract_blocks_from_page(page_num)

            math_blocks = sum(1 for b in blocks if b.get("has_math", False))
            total_math_blocks += math_blocks
            
            if math_blocks > 0:
                print(f"  ✓ Found {math_blocks} blocks with mathematical content")

            chunk = {
                "page": page_num,
                "text": page_text,
                "images": images,
                "blocks": blocks,
                "has_images": len(images) > 0,
                "image_count": len(images),
                "has_math": math_blocks > 0,
                "math_block_count": math_blocks
            }
            
            all_chunks.append(chunk)
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction Summary:")
        print(f"  - Total pages: {len(self.doc)}")
        print(f"  - Total images/diagrams: {total_images}")
        print(f"  - Blocks with mathematical content: {total_math_blocks}")
        print(f"{'='*60}")
        
        return all_chunks
