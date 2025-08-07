# PDF processor with caching and page-aware chunking
import pickle
from pathlib import Path
from docling.document_converter import DocumentConverter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from page_aware_chunker import chunk_pdf_with_pages

class PdfProcessor:
    def __init__(self, converter="docling"):
        """Initialize PDF converter with pre-loaded model"""
        if converter not in ["marker", "docling"]:
            raise ValueError("Invalid converter. Please choose 'marker' or 'docling'.")
        
        self.converter_type = converter
        if converter == "marker":
            self.converter = PdfConverter(artifact_dict=create_model_dict())
        elif converter == "docling":
            self.converter = DocumentConverter()
            self.converter.initialize_pipeline("pdf")
    
    def process_document(self,
                         doc_path,
                         save_path=None, 
                         chunk_size=None, 
                         chunk_overlap=None,
                         embeddings_path=None,
                         model_name=None):
        """Process document using PKL-based caching instead of .md files"""
        # Import here to avoid circular imports
        from config.session_state import session_data
        from models.progress_tracking import progress_tracker
        
        # Check if path is to a single file or a directory
        # Process a single file or all files in a directory
        all_texts = []
        all_page_chunks = []
        
        # Handle URLs and file paths
        if isinstance(doc_path, str) and (doc_path.startswith('http://') or doc_path.startswith('https://')):
            files = [doc_path]  # URL as string
        else:
            # Use the file directly if it's a single file, otherwise get all PDFs in directory
            files = [doc_path] if doc_path.is_file() else list(doc_path.glob("*.pdf"))
        
        # Process each file
        for i, file in enumerate(files):
            # Get file name for URL or path
            if isinstance(file, str) and (file.startswith('http://') or file.startswith('https://')):
                file_stem = file.split('/')[-1].replace('.pdf', '')
            else:
                file_stem = file.stem
            
            # Use provided chunk_size and chunk_overlap, or fall back to session defaults
            if chunk_size is None:
                chunk_size = session_data['bi_encoder_config']['chunk_size']
            if chunk_overlap is None:
                chunk_overlap = session_data['bi_encoder_config']['chunk_overlap']
            
            # Check if PKL files already exist for this document with current configuration
            skip_processing = False
            if embeddings_path and model_name:
                # Build the configuration-specific PKL path
                pkl_path = Path(embeddings_path) / model_name.split("/")[-1] / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}" / file_stem
                if pkl_path.exists():
                    chunk_files = list(pkl_path.glob("chunk_*.pkl"))
                    if chunk_files:
                        progress_tracker.log_message(f"Skipping {file_stem} - PKL embeddings already exist for current config ({len(chunk_files)} chunks)")
                        # Load page chunks from existing PKL files
                        page_chunks = []
                        for chunk_file in sorted(chunk_files):
                            with open(chunk_file, "rb") as f:
                                chunk_data = pickle.load(f)
                                page_chunks.append({
                                    'text': chunk_data['text'],
                                    'page': chunk_data['page']
                                })
                        all_page_chunks.append(page_chunks)
                        # Reconstruct full text from chunks
                        full_text = '\n\n'.join([chunk['text'] for chunk in page_chunks])
                        all_texts.append(full_text)
                        skip_processing = True
            
            if skip_processing:
                continue
            
            progress_tracker.log_message(f"Processing {file_stem} ({i+1}/{len(files)})")
            
            if self.converter_type == "marker":
                rendered_doc = self.converter(str(file))
                text, _, images = text_from_rendered(rendered_doc)
                all_texts.append(text)
                
                # For marker, we still need to use page-aware chunking
                progress_tracker.log_message(f"Creating page-aware chunks for {file_stem}")
                page_chunks, _ = chunk_pdf_with_pages(str(file), chunk_size, chunk_overlap)
                all_page_chunks.append(page_chunks)
                
            elif self.converter_type == "docling":
                # Use page-aware chunking directly for docling
                progress_tracker.log_message(f"Creating page-aware chunks for {file_stem}")
                page_chunks, full_text = chunk_pdf_with_pages(str(file), chunk_size, chunk_overlap)
                all_texts.append(full_text)
                all_page_chunks.append(page_chunks)
        
        # For single files, return the text and page chunks as lists with one item
        is_single_file = (isinstance(doc_path, str) and (doc_path.startswith('http://') or doc_path.startswith('https://'))) or \
                        (hasattr(doc_path, 'is_file') and doc_path.is_file())
        
        if is_single_file:
            return all_texts[0:1], all_page_chunks[0:1]
        else:
            return all_texts, all_page_chunks