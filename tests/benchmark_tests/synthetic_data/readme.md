1. Install Python Dependencies:

pip install pytesseract pdf2image PyPDF2 docx transformers torch

How It Works: 
- The script will automatically create a processed_files directory in the same folder as the script, if it does not already exist.
- You provide a list of file names you want to summarize (e.g., file1.pdf, file2.docx).
- The script copies the files to the processed_files directory, processes them to extract text, and generates a summary for each file.
- The summary is saved in the same processed_files directory with the name <original_file_name>_summary.txt.

2.Place your files (PDF, DOCX, TXT, or JSON) in the same directory as the script.

3. Run the script with the names of the files you want to summarize. For example:
python synthetic_data.py file1.pdf file2.docx file3.txt

After the script finishes, you'll find:
- Your original files in the processed_files directory.
- Summarized versions of each file as <original_file_name>_summary.txt.