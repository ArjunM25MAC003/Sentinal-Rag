"""
PDF processing module for extracting and indexing text from PDFs.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    logger.warning("PyMuPDF (fitz) is not installed. PDF extraction will fail.")

try:
    import faiss
except ImportError:
    logger.warning("faiss is not installed. FAISS indexing will fail.")
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence_transformers is not installed. Embeddings will fail.")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Uses pymupdf (fitz) to extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks.
    
    Args:
        text (str): The full text to chunk.
        chunk_size (int): The target size of each chunk.
        overlap (int): The number of characters to overlap between chunks.
        source_file (str): The original filename.
        
    Returns:
        List[dict]: List of chunk dictionaries.
    """
    chunks = []
    if not text:
        return chunks
        
    start = 0
    text_length = len(text)
    chunk_index = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text_content = text[start:end]
        
        chunks.append({
            'text': chunk_text_content,
            'chunk_index': chunk_index,
            'source_file': source_file,
            'source_type': 'pdf'
        })
        
        chunk_index += 1
        # Move forward, adjusting for overlap
        start += (chunk_size - overlap)
        
    return chunks

def build_faiss_index(chunks: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Builds a FAISS inner-product index from chunks using sentence-transformers.
    
    Args:
        chunks (List[dict]): The chunks to index.
        model_name (str): Sentence transformer model name.
        
    Returns:
        Tuple[faiss.IndexFlatIP, List[dict]]: The FAISS index and the corresponding chunks.
    """
    if not chunks:
        return None, []
        
    try:
        model = SentenceTransformer(model_name)
        texts = [chunk['text'] for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, chunks
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        return None, chunks

def search_faiss_index(query: str, faiss_index: Any, chunks: List[Dict[str, Any]], model: Any, top_k: int = 3, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index and returns matching chunks with scores.
    
    Args:
        query (str): The search query.
        faiss_index: The FAISS index object.
        chunks (List[dict]): The list of chunks corresponding to the index.
        model: The SentenceTransformer model.
        top_k (int): Number of top results to return.
        threshold (float): Minimum similarity score threshold.
        
    Returns:
        List[dict]: List of matching chunks with similarity scores.
    """
    if faiss_index is None or not chunks or model is None:
        return []
        
    try:
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(chunks):
                score = float(scores[0][i])
                if score >= threshold:
                    chunk = chunks[idx].copy()
                    chunk['score'] = score
                    results.append(chunk)
                    
        return results
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        return []

class PDFKnowledgeBase:
    """
    A class that manages a knowledge base built from PDF documents.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the PDFKnowledgeBase.
        
        Args:
            model_name (str): The sentence transformer model to use.
        """
        self.model_name = model_name
        self.chunks: List[Dict[str, Any]] = []
        self.index = None
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.model = None

    def add_pdf(self, pdf_path: str) -> None:
        """
        Extracts, chunks, and indexes a single PDF.
        
        Args:
            pdf_path (str): Path to the PDF file.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return
            
        text = extract_text_from_pdf(pdf_path)
        if text:
            filename = os.path.basename(pdf_path)
            new_chunks = chunk_text(text, source_file=filename)
            self._update_index(new_chunks)

    def add_pdfs(self, pdf_paths: List[str]) -> None:
        """
        Processes and indexes multiple PDFs.
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files.
        """
        for pdf_path in pdf_paths:
            self.add_pdf(pdf_path)

    def _update_index(self, new_chunks: List[Dict[str, Any]]) -> None:
        """
        Helper to append new chunks and rebuild the FAISS index.
        
        Args:
            new_chunks (List[dict]): New chunks to add.
        """
        if not new_chunks or self.model is None:
            return
            
        self.chunks.extend(new_chunks)
        # Rebuilding index from scratch with all chunks
        # A more advanced implementation would use faiss.IndexFlatIP.add incrementally,
        # but rebuilding is simpler and fine for modest sizes.
        self.index, _ = build_faiss_index(self.chunks, self.model_name)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the indexed PDFs.
        
        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.
            
        Returns:
            List[dict]: List of search results.
        """
        if self.is_empty():
            return []
            
        return search_faiss_index(query, self.index, self.chunks, self.model, top_k=top_k)

    def is_empty(self) -> bool:
        """
        Checks if any PDFs are indexed.
        
        Returns:
            bool: True if no chunks are indexed, False otherwise.
        """
        return len(self.chunks) == 0 or self.index is None
