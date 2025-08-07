# Content processing utilities for HTML parsing and formatting
import re
import os
import pickle
import nltk
from pathlib import Path
from dash import html
import dash_bootstrap_components as dbc
from rank_bm25 import BM25Okapi

def parse_html_content(content):
    """Parse content and convert to properly formatted Dash components"""
    if not content:
        return []
    
    # Split content into lines to preserve line breaks
    lines = content.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            # Empty line - add a line break
            result.append(html.Br())
            continue
        
        # Process the line for links and formatting
        line_components = []
        
        # Enhanced regex to find HTML links
        link_pattern = r'<a href=[\'"]([^\'"]*)[\'"]>([^<]*)</a>'
        
        last_end = 0
        for match in re.finditer(link_pattern, line):
            # Add text before the link
            if match.start() > last_end:
                text_before = line[last_end:match.start()]
                if text_before:
                    line_components.append(text_before)
            
            # Add the link as html.A component
            href = match.group(1)
            link_text = match.group(2)
            line_components.append(html.A(link_text, href=href, target="_blank", 
                                        style={"color": "#0066cc", "textDecoration": "underline"}))
            
            last_end = match.end()
        
        # Add remaining text after the last link
        if last_end < len(line):
            remaining_text = line[last_end:]
            if remaining_text:
                line_components.append(remaining_text)
        
        # If no links were found, just add the whole line as text
        if not line_components:
            line_components.append(line)
        
        # Add the line components
        result.extend(line_components)
        
        # Add line break after each line (except the last one)
        if i < len(lines) - 1:
            result.append(html.Br())
    
    return result

def enhance_response_with_links(response_content, top_docs, current_project):
    """Add project-specific PDF links to LLM response"""
    if not top_docs or not current_project:
        return response_content
    
    # Extract document names and page numbers from top_docs
    context_ids = []
    context_pages = []
    
    for doc in top_docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            # Extract document name (remove extension)
            if 'source' in doc.metadata:
                doc_name = Path(doc.metadata['source']).stem
            elif 'document_name' in doc.metadata:
                doc_name = doc.metadata['document_name']
            else:
                doc_name = getattr(doc, 'document_name', 'unknown')
            
            # Extract page number
            page_num = doc.metadata.get('page_number', 1)
            
            context_ids.append(doc_name)
            context_pages.append(page_num)
        else:
            # Fallback for documents without proper metadata
            doc_name = getattr(doc, 'document_name', f'doc_{len(context_ids)}')
            page_num = getattr(doc, 'page_number', 1)
            context_ids.append(doc_name)
            context_pages.append(page_num)
    
    # Handle both old and new document reference formats
    for i, doc_name in enumerate(context_ids):
        # Old format: DOCUMENT1, DOCUMENT2, etc.
        response_content = response_content.replace(f"DOCUMENT{i+1}",
                                                    f"<a href='docs_pdf/{current_project}/{doc_name}.pdf#page={context_pages[i]}'>[{i+1}]</a>")
    
    # New format: <document_id>X</document_id>
    doc_id_pattern = r'<document_id>(\d+)</document_id>'
    def replace_doc_id(match):
        doc_num = int(match.group(1))
        if 1 <= doc_num <= len(context_ids):
            doc_name = context_ids[doc_num-1]
            return f"<a href='docs_pdf/{current_project}/{doc_name}.pdf#page={context_pages[doc_num-1]}'>[{doc_num}]</a>"
        return match.group(0)  # Return original if invalid
    
    response_content = re.sub(doc_id_pattern, replace_doc_id, response_content)
    
    return response_content

def create_error_alert(message):
    """Create an error alert component"""
    return dbc.Alert(
        message,
        color="danger",
        dismissable=True,
        className="mb-3"
    )

def save_processed_documents(doc_texts, doc_ids, save_dir):
    """Save processed documents to disk"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for i, (text, doc_id) in enumerate(zip(doc_texts, doc_ids)):
        file_path = save_path / f"{doc_id}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

def load_processed_documents(load_dir):
    """Load processed documents from disk"""
    load_path = Path(load_dir)
    if not load_path.exists():
        return [], []
    
    doc_texts = []
    doc_ids = []
    
    for file_path in load_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        doc_texts.append(text)
        doc_ids.append(file_path.stem)
    
    return doc_texts, doc_ids

def get_bm25_path(base_dir, project_name, model_name, chunk_size, chunk_overlap):
    """Get the path for BM25 corpus file"""
    model_name_clean = model_name.split("/")[-1] if "/" in model_name else model_name
    bm25_dir = Path(base_dir) / "projects" / project_name / "embeddings" / model_name_clean / f"chunk_size_{chunk_size}" / f"chunk_overlap_{chunk_overlap}"
    return bm25_dir / "bm25_corpus.pkl"

def create_bm25_corpus(embeddings, bm25_path):
    """Create and save BM25 corpus"""
    if not embeddings:
        return None, None
    
    # Extract texts from embeddings
    texts = []
    for emb in embeddings:
        if hasattr(emb, 'page_content'):
            texts.append(emb.page_content)
        elif isinstance(emb, dict) and 'text' in emb:
            texts.append(emb['text'])
        else:
            texts.append(str(emb))
    
    # Tokenize texts for BM25
    try:
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
        tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
    except:
        # Fallback to simple split if NLTK fails
        tokenized_corpus = [text.lower().split() for text in texts]
    
    # Create BM25 object
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save to disk
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, 'wb') as f:
        pickle.dump((bm25, tokenized_corpus), f)
    
    return bm25, tokenized_corpus

def load_bm25_corpus(bm25_path):
    """Load BM25 corpus from disk"""
    if not bm25_path.exists():
        return None, None
    
    try:
        with open(bm25_path, 'rb') as f:
            bm25, tokenized_corpus = pickle.load(f)
        return bm25, tokenized_corpus
    except:
        return None, None