"""
Summarization Test Module

Evaluates an LLM's ability to generate concise and accurate summaries by:
1. Loading test cases from various file formats (JSON, PDF, DOCX, TXT)
2. Generating summaries using a CPU-friendly summarization model
3. Creating separate summary text files for each input file
"""

import json
import asyncio
from pathlib import Path
import sys

# For reading various file formats
from docx import Document
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Transformers for our adapter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deepeval.models import DeepEvalBaseLLM

# ------------------------------------------------------------------------------
# Custom Adapter: DistilBARTSummarizer
# ------------------------------------------------------------------------------
class DistilBARTSummarizer(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # Create a summarization pipeline on CPU
        self.pipe = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu"  # Force CPU usage
        )
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        # Define maximum input token length for the model
        max_input_length = 1024
        # Tokenize the prompt
        encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(encoded) > max_input_length:
            # Truncate the input tokens if too long
            encoded = encoded[:max_input_length]
            prompt = self.tokenizer.decode(encoded, skip_special_tokens=True)
            print("Input truncated due to length.")
        result = self.pipe(prompt, max_length=max_tokens, **kwargs)
        return result[0]['summary_text']
    
    def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def load_model(self):
        return self.model

# ------------------------------------------------------------------------------
# File Reading Helper Function
# ------------------------------------------------------------------------------
def read_file(filepath: str) -> str:
    """
    Reads and returns the content of a file.
    Supports .txt, .docx, .pdf, and .json.
    """
    try:
        if filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.lower().endswith(".docx"):
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filepath.lower().endswith(".pdf"):
            with open(filepath, "rb") as f:
                pdf_reader = PdfReader(f)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            if not text.strip():
                # Use OCR as fallback if text extraction fails.
                images = convert_from_path(filepath)
                text = "\n".join([pytesseract.image_to_string(image) for image in images])
            return text
        elif filepath.lower().endswith(".json"):
            # If JSON contains a structured test case (e.g., with an "input" key), you can adjust this logic.
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # If data is a dict with an "input" key, return that.
                    if isinstance(data, dict) and "input" in data:
                        return data["input"]
                    else:
                        # Otherwise, return the pretty-printed JSON string.
                        return json.dumps(data, indent=2)
                except Exception as e:
                    return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return ""

# ------------------------------------------------------------------------------
# Simulated Chat API Call for Summarization
# ------------------------------------------------------------------------------
async def get_chat_response(input_text: str) -> str:
    """
    Simulate an asynchronous chat API call to get a summary.
    Here we directly use our DistilBARTSummarizer adapter.
    Replace this with your actual API call if needed.
    """
    summarizer = DistilBARTSummarizer()
    # Run generation in an executor to avoid blocking the event loop.
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, summarizer.generate, input_text, 150)
    return summary

# ------------------------------------------------------------------------------
# Main Entry Point: Process Multiple Files
# ------------------------------------------------------------------------------
async def main():
    if len(sys.argv) < 2:
        print("Usage: python synthetic_data.py <file1> [<file2> ...]")
        return

    file_paths = sys.argv[1:]
    for fp in file_paths:
        fp_path = Path(fp)
        print(f"\nProcessing file: {fp_path}")
        content = read_file(str(fp_path))
        if not content:
            print(f"No content found in {fp_path}. Skipping.")
            continue
        
        summary = await get_chat_response(content)
        print("Generated Summary:")
        print(summary)
        
        # Write the summary to a new text file in the same directory.
        output_path = fp_path.with_name(fp_path.stem + "_summary.txt")
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(summary)
        print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

