import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Tuple, Any
import argparse
from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
import io
from transformers import AutoModel, AutoProcessor
from pdfProcess import extract_pdf_content


@dataclass
class PDFChunk:
    """Represents a logical section/chunk of a PDF document"""
    id: str
    title: str
    text: str
    page_range: Tuple[int, int]
    images: List[str]  # Paths to extracted images
    text_embedding: Any = None
    image_embeddings: List[Any] = None


def load_embedding_model():
    """Load a multimodal embedding model that can handle both text and images"""
    # Example using a multimodal model - replace with your preferred model
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    processor = AutoProcessor.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    return model, processor


def generate_embeddings(chunks, model, processor):
    """Generate embeddings for text chunks and images"""
    # Set model to evaluation mode
    model.eval()

    for chunk in chunks:
        # Generate text embeddings
        text_inputs = processor(chunk.text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            text_outputs = model(**text_inputs)
            # Use the [CLS] token embedding or mean pooling depending on the model
            chunk.text_embedding = text_outputs.last_hidden_state[:, 0, :].numpy()

        # Generate embeddings for each image in the chunk
        chunk.image_embeddings = []
        for img_path in chunk.images:
            try:
                # Load image
                image = Image.open(img_path)
                # Process image for the model
                image_inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    image_outputs = model(**image_inputs)
                    # Extract image embedding
                    image_embedding = image_outputs.last_hidden_state[:, 0, :].numpy()
                    chunk.image_embeddings.append(image_embedding)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Add a placeholder embedding
                chunk.image_embeddings.append(None)

    return chunks


def create_vector_dataframe(chunks):
    """Convert chunks with embeddings to a DataFrame for storage/retrieval"""
    records = []

    # Add text chunks
    for chunk in chunks:
        record = {
            'id': chunk.id,
            'title': chunk.title,
            'content': chunk.text,
            'type': 'text',
            'page_range': f"{chunk.page_range[0] + 1}-{chunk.page_range[1] + 1}",
            'vector': chunk.text_embedding
        }
        records.append(record)

        # Add image records
        for i, (img_path, img_embedding) in enumerate(zip(chunk.images, chunk.image_embeddings)):
            if img_embedding is not None:
                img_record = {
                    'id': f"{chunk.id}_img_{i}",
                    'title': f"Image {i + 1} from {chunk.title}",
                    'content': img_path,  # Store the path to the image
                    'type': 'image',
                    'page_range': f"{chunk.page_range[0] + 1}-{chunk.page_range[1] + 1}",
                    'vector': img_embedding,
                    'parent_chunk': chunk.id
                }
                records.append(img_record)

    # Create DataFrame
    df = pd.DataFrame(records)
    return df


def extract_pdf_content_with_embeddings(pdf_path, output_dir):
    """Extract content from PDF, chunk it, extract images, and generate embeddings"""
    # Load the embedding model
    model, processor = load_embedding_model()

    # Extract content as before
    chunks = extract_pdf_content(pdf_path, output_dir)

    # Generate embeddings
    chunks_with_embeddings = generate_embeddings(chunks, model, processor)

    # Create a DataFrame with vectors
    vector_df = create_vector_dataframe(chunks_with_embeddings)

    # Save the DataFrame
    vector_df_path = os.path.join(output_dir, "vector_store.pkl")
    vector_df.to_pickle(vector_df_path)

    return vector_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract content and images from a PDF file and generate vector embeddings")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default="pdf_extracted", help="Output directory")
    parser.add_argument("--embeddings", "-e", action="store_true", help="Generate embeddings for text and images")
    args = parser.parse_args()

    if args.embeddings:
        vector_df = extract_pdf_content_with_embeddings(args.pdf_path, args.output)
        print(f"Generated embeddings for {len(vector_df)} items (text chunks and images)")
        print(f"Vector store saved to {args.output}/vector_store.pkl")
    else:
        chunks = extract_pdf_content(args.pdf_path, args.output)
        print(f"PDF processed successfully!")
        print(f"Found {len(chunks)} logical sections")
        total_images = sum(len(chunk.images) for chunk in chunks)
        print(f"Extracted {total_images} images")

    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()