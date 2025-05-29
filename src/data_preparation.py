"""
data_preparation.py
===================
This script prepares your domain-specific dataset for fine-tuning the LLM.
It reads all .txt and .pdf files from the 'data' directory, extracts and cleans text,
splits the text into smaller chunks, and creates a Hugging Face Dataset.

Key Steps:
  - Read .txt files using standard file I/O.
  - Extract text from .pdf files using PyPDF2.
  - Clean text by removing extra whitespace.
  - Split long texts into chunks of a specified maximum character length.
  - Create a dataset from the list of text chunks.
"""

import os
import re
import PyPDF2
from datasets import Dataset


def read_txt_file(filepath: str) -> str:
    """
    Reads a .txt file and returns its content as a single string.

    Args:
        filepath (str): Path to the .txt file.

    Returns:
        str: The content of the file.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf_file(filepath: str) -> str:
    """
    Reads a .pdf file using PyPDF2 and returns its full text.

    Args:
        filepath (str): Path to the .pdf file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(filepath, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return text


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace and line breaks.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text_into_chunks(text: str, max_chars: int = 500) -> list:
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The full text to split.
        max_chars (int): Maximum number of characters per chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


def prepare_document_dataset(data_dir: str = None, max_chars: int = 500) -> Dataset:
    """
    Processes all documents in the specified data directory, extracts their text,
    cleans and splits them into chunks, and returns a Hugging Face Dataset.

    Args:
        data_dir (str): Directory where the document files are stored. If None, the default
                        path is determined relative to this script.
        max_chars (int): Maximum number of characters per text chunk.

    Returns:
        Dataset: A dataset with one column "text" containing chunks.
    """
    # Determine the absolute path to the data directory if not provided.
    if data_dir is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(current_dir, "..", "data")

    all_chunks = []

    # Iterate over all files in the directory.
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        # Process .txt files.
        if filename.lower().endswith(".txt"):
            print(f"Reading text file: {filename}")
            content = read_txt_file(filepath)
        # Process .pdf files.
        elif filename.lower().endswith(".pdf"):
            print(f"Reading PDF file: {filename}")
            content = read_pdf_file(filepath)
        else:
            continue

        # Clean and split the text.
        content = clean_text(content)
        chunks = split_text_into_chunks(content, max_chars=max_chars)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    # Create a dataset with a single "text" column.
    dataset = Dataset.from_dict({"text": all_chunks})
    return dataset


if __name__ == "__main__":
    # Prepare the dataset and display some information.
    dataset = prepare_document_dataset(data_dir=None, max_chars=500)
    print(f"Prepared dataset with {len(dataset)} samples.")
    # Optionally, preview the first sample.
    print(dataset[0])
