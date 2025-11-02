from typing import List, Dict, Any
import re
class AcademicChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def is_formula_header(self, text: str) -> bool:
        formula_patterns = [
            r'equation\s+\d+',
            r'formula\s+\d+',
            r'theorem\s+\d+',
            r'example\s+\d+',
            r'Fig\.\s*\d+\.\d+',
            r'solution:',
            r'^\s*\d+\.\d+',    # Section numbers
            r'tan\s*\d+°',
            r'sin\s*\d+°',
            r'cos\s*\d+°',
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def extract_figure_references(self, text: str) -> List[str]:
        pattern = r'Fig\.\s*\d+\.\d+'
        return re.findall(pattern, text)
    
    def split_into_sentences(self, text: str) -> List[str]:

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|\n)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        if len(text) < self.chunk_size:
            return [{
                "text": text.strip(),
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "is_single_chunk": True
                }
            }]
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if this is a formula header - keep with next sentence
            is_header = self.is_formula_header(sentence)
            
            # Check if adding this sentence would exceed chunk size
            test_length = len(current_chunk) + len(sentence) + 1
            
            if test_length > self.chunk_size and current_chunk and not is_header:
                # Saving current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "sentence_count": len(current_sentences),
                        "has_formulas": self.is_formula_header(current_chunk)
                    }
                })
                
                # Starting new chunk with overlap
                overlap_size = min(2, len(current_sentences))
                overlap_text = " ".join(current_sentences[-overlap_size:]) if overlap_size > 0 else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_sentences = current_sentences[-overlap_size:] + [sentence] if overlap_size > 0 else [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Adding final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_sentences),
                    "has_formulas": self.is_formula_header(current_chunk)
                }
            })
        
        return chunks
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        
        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]
            images = page_data["images"]
            has_math = page_data.get("has_math", False)
            
            text_chunks = self.chunk_text(
                text,
                {
                    "page": page_num,
                    "source": "text",
                    "has_images": page_data["has_images"],
                    "has_math": has_math
                }
            )
            all_chunks.extend(text_chunks)
            
            for img in images:
                # Extracts surrounding text context for image
                context_text = f"This page discusses trigonometry applications including heights, distances, angles of elevation and depression."
                
                img_chunk = {
                    "text": f"[DIAGRAM on Page {page_num + 1}] Figure/Diagram showing trigonometric concepts. Image file: {img['filename']}. {context_text} The diagram illustrates geometric relationships, triangles, angles, or worked examples related to the text on this page.",
                    "metadata": {
                        "page": page_num,
                        "source": "image",
                        "image_path": img["path"],
                        "image_filename": img["filename"],
                        "image_index": img["index"],
                        "chunk_type": "diagram"
                    }
                }
                all_chunks.append(img_chunk)
        
        return all_chunks