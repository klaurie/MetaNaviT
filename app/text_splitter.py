from typing import List
import re

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter with configurable chunk size and overlap.
        
        Args:
            chunk_size (int): Maximum size of each text chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks of approximately equal size.
        
        Args:
            text (str): The input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Clean the text
        text = self._clean_text(text)
        
        # If text is shorter than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # If this is the last chunk, just add it
                chunks.append(text[start:])
                break
                
            # Try to find a natural break point
            split_point = self._find_split_point(text[start:end])
            
            if split_point:
                chunks.append(text[start:start + split_point])
                start = start + split_point - self.chunk_overlap
            else:
                # If no natural break point, just split at chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean the input text by removing extra whitespace."""
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _find_split_point(self, text: str) -> int:
        """
        Find a natural split point in the text (end of sentence or paragraph).
        Returns the index of the split point or None if no good split point found.
        """
        # Try to split at paragraph
        paragraph_split = text.rfind('\n')
        if paragraph_split > self.chunk_size * 0.5:
            return paragraph_split
            
        # Try to split at sentence
        sentence_splits = [
            text.rfind('. '),
            text.rfind('? '),
            text.rfind('! '),
            text.rfind('.\n'),
            text.rfind('?\n'),
            text.rfind('!\n')
        ]
        
        best_split = max(split for split in sentence_splits if split > self.chunk_size * 0.5)
        
        if best_split > 0:
            return best_split + 2  # Include the punctuation and space
            
        # If no good natural break point found, return None
        return None