import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Tuple, Any
import argparse
from dataclasses import dataclass


@dataclass
class PDFChunk:
    """Represents a logical section/chunk of a PDF document"""
    id: str
    title: str
    text: str
    page_range: Tuple[int, int]
    images: List[str]  # Paths to extracted images

def process_page_one(lines:list):
    found_abs = False
    result_list = [lines[0]]
    for line in lines:
        if found_abs or line == "Abstract":
            result_list.append(line)
            found_abs = True
    return result_list


def extract_pdf_content(pdf_path: str, output_dir: str) -> List[PDFChunk]:
    """
    Extract content from PDF, chunk it into sections, and extract images

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images and text

    Returns:
        List of PDFChunk objects representing logical sections of the document
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Open the PDF
    doc = fitz.open(pdf_path)

    # Extract potential section headers and their page numbers
    headers = []
    full_text = ""

    #处理掉那群名字 和 school 但是要保留 title
    for page_num, page in enumerate(doc):
        # Get text from the page
        text = page.get_text()
        full_text += text
        # Look for potential headers (lines with fewer words, possibly in larger font)
        lines = text.split('\n')
        if page_num == 1:
            lines = process_page_one(lines)
        for line in lines:
            line = line.strip()
            # Simple heuristic: potential headers are short and not ending with punctuation
            if 2 <= len(line.split()) <= 7 and line and not line[-1] in '.,:;?!)/0123456789':
                # Check if the line has larger font than average text
                # This is a simplification; in practice you might want to analyze font sizes
                headers.append((line, page_num))

    # Identify document sections based on headers
    chunks = []
    current_chunk_text = ""
    current_chunk_start_page = 0
    current_chunk_title = "Introduction"  # Default title for the first section
    current_chunk_images = []

    # Process images in the document
    image_count = 0
    # Header 到底是什么作用
    # chunk size 如何implement
    # Overlap 一样的 implement
    #
    for page_num, page in enumerate(doc):
        # Check if we're at a new section
        for header, header_page in headers:
            if header_page == page_num and len(current_chunk_text) > 0:
                # Save the current chunk before starting a new one
                chunk_id = f"chunk_{len(chunks)}"
                chunks.append(PDFChunk(
                    id=chunk_id,
                    title=current_chunk_title,
                    text=current_chunk_text,
                    page_range=(current_chunk_start_page, page_num - 1),
                    images=current_chunk_images.copy()
                ))

                # Start a new chunk
                current_chunk_text = ""
                current_chunk_start_page = page_num
                current_chunk_title = header
                current_chunk_images = []

        # Add page text to current chunk
        current_chunk_text += page.get_text()

        # Extract images from the page
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # Image reference in the PDF

            # Extract the image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Save the image
            image_filename = f"image_{image_count:03d}.{image_ext}"
            image_path = os.path.join(images_dir, image_filename)

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            # Add image to current chunk
            current_chunk_images.append(image_path)
            image_count += 1

    # Add the last chunk
    if current_chunk_text:
        chunk_id = f"chunk_{len(chunks)}"
        chunks.append(PDFChunk(
            id=chunk_id,
            title=current_chunk_title,
            text=current_chunk_text,
            page_range=(current_chunk_start_page, len(doc) - 1),
            images=current_chunk_images
        ))

    # Save chunks to text files
    for chunk in chunks:
        chunk_filename = f"{chunk.id}.txt"
        chunk_path = os.path.join(output_dir, chunk_filename)

        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {chunk.title}\n")
            f.write(f"Pages: {chunk.page_range[0] + 1}-{chunk.page_range[1] + 1}\n")
            f.write(f"Images: {', '.join(os.path.basename(img) for img in chunk.images)}\n\n")
            f.write(chunk.text)

    return chunks


def main():
    # parser = argparse.ArgumentParser(description="Extract content and images from a PDF file")
    # parser.add_argument("pdf_path", help="Path to the PDF file")
    # parser.add_argument("--output", "-o", default="pdf_extracted", help="Output directory")
    # args = parser.parse_args()
    pdf_path = "/Users/henry/Documents/Project/PracticeProject/Multimodal-Search/data/PDF/2410.02536v2.pdf"
    chunk_output = "/Users/henry/Documents/Project/PracticeProject/Multimodal-Search/data/pdfOutput"

    chunks = extract_pdf_content(pdf_path, chunk_output)

    print(f"PDF processed successfully!")
    print(f"Found {len(chunks)} logical sections")

    # Count total images
    total_images = sum(len(chunk.images) for chunk in chunks)
    print(f"Extracted {total_images} images")

    print(f"Results saved to {chunk_output}/")


if __name__ == "__main__":
    main()