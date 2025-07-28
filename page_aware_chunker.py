from docling.document_converter import DocumentConverter
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pdf_with_pages(pdf_path, chunk_size=2048, chunk_overlap=128):
    """
    Process a PDF and chunk it while preserving page information.
    
    Args:
        pdf_path (str): Path to PDF file
        chunk_size (int): Maximum size of each chunk  
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of dictionaries with chunk text and page info
    """
    # Convert PDF using docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    
    print(f"Processing document with {len(doc.pages)} pages")
    
    # Build character position to page mapping
    char_to_page = {}
    full_text = ""
    current_pos = 0
    
    for item_tuple in doc.iterate_items():
        element = item_tuple[0]
        
        # Try to extract text from different element types
        element_text = None
        page_no = None
        
        # Check for direct text attribute
        if hasattr(element, 'text') and element.text:
            element_text = element.text
        # Check for table-specific attributes (with proper doc argument)
        elif hasattr(element, 'export_to_markdown'):
            try:
                element_text = element.export_to_markdown(doc)
            except:
                try:
                    # Fallback without doc argument if needed
                    element_text = element.export_to_markdown()
                except:
                    pass
        elif hasattr(element, 'export_to_text'):
            try:
                element_text = element.export_to_text()
            except:
                pass
        # Check for picture captions
        elif hasattr(element, 'caption_text') and element.caption_text:
            element_text = f"[Image Caption: {element.caption_text}]"
        # Check for other text representations
        elif hasattr(element, 'get_text'):
            try:
                element_text = element.get_text()
            except:
                pass
        
        # Try to get page information
        if hasattr(element, 'prov') and element.prov:
            for prov in element.prov:
                if hasattr(prov, 'page_no'):
                    page_no = prov.page_no
                    break
        
        if element_text and page_no is not None:
            # Filter out image placeholder text
            if "🖼️❌ Image not available" in element_text:
                continue
                
            element_text = element_text + "\n"
            
            # Map each character position to its page
            for i in range(len(element_text)):
                char_to_page[current_pos + i] = page_no
            
            full_text += element_text
            current_pos += len(element_text)
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    
    # Determine page for each chunk using direct text search
    chunks_with_pages = []
    last_found_pos = 0  # Track position to avoid finding same text twice
    
    for i, chunk in enumerate(chunks):
        # Find the actual position of this chunk in the full text
        # Search starting from after the last found position
        search_text = chunk[:min(200, len(chunk))]  # First 200 chars as search key
        
        chunk_start = full_text.find(search_text, last_found_pos)
        if chunk_start == -1:
            # Fallback: try with first 100 chars if 200 didn't work
            search_text = chunk[:min(100, len(chunk))]
            chunk_start = full_text.find(search_text, last_found_pos)
        
        if chunk_start == -1:
            # Last resort: try first 50 chars
            search_text = chunk[:min(50, len(chunk))]
            chunk_start = full_text.find(search_text, last_found_pos)
            
        if chunk_start == -1:
            print(f"Warning: Could not find chunk {i} in full text, using position estimate")
            # Fallback: estimate position
            if i == 0:
                chunk_start = 0
            else:
                # Estimate based on previous chunk position
                prev_start = chunks_with_pages[i-1].get('_debug_start', 0)
                prev_length = len(chunks[i-1])
                # Assume some overlap, but don't rely on exact chunk_overlap value
                estimated_overlap = min(chunk_overlap, prev_length // 4)  # Conservative estimate
                chunk_start = prev_start + prev_length - estimated_overlap
        
        chunk_end = chunk_start + len(chunk)
        
        # Count characters from each page in this chunk
        page_counts = defaultdict(int)
        for pos in range(chunk_start, min(chunk_end, len(full_text))):
            if pos in char_to_page:
                page_counts[char_to_page[pos]] += 1
        
        # Primary page is the one with most characters
        primary_page = max(page_counts.items(), key=lambda x: x[1])[0] if page_counts else 1
        
        chunks_with_pages.append({
            'text': chunk,
            'page': primary_page,
            'chunk_index': i,
            '_debug_start': chunk_start,  # For debugging
            '_debug_end': chunk_end
        })
        
        # Update last found position for next search
        if chunk_start != -1:
            last_found_pos = chunk_start + len(search_text)
    
    return chunks_with_pages, full_text

# Example usage
if __name__ == "__main__":
    pdf_path = "https://arxiv.org/pdf/2408.09869"  # Replace with your PDF path
    
    chunks, full_text = chunk_pdf_with_pages(pdf_path)
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Debug: Check total text length and chunk sizes
    total_chars = len(full_text)
    avg_chunk_size = sum(len(chunk['text']) for chunk in chunks) / len(chunks)
    
    print(f"Total text length: {total_chars} characters")
    print(f"Average chunk size: {avg_chunk_size:.1f} characters")
    
    # Show first few chunks
    print("\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk['chunk_index']} (Page {chunk['page']}) [{len(chunk['text'])} chars] ---")
        print(f"{chunk['text'][:200]}...")
    
    # Show statistics about content types
    table_count = 0
    for chunk in chunks:
        lines = chunk['text'].split('\n')
        for j, line in enumerate(lines):
            if '|' in line and j + 1 < len(lines):
                next_line = lines[j + 1]
                if '|' in next_line and ('-' in next_line or ':' in next_line):
                    table_count += 1
                    break
    
    print(f"\nContent summary: {table_count} chunks contain tables")
    
    # Debug: Show page distribution and positions for verification
    page_distribution = {}
    for chunk in chunks:
        page = chunk['page']
        if page not in page_distribution:
            page_distribution[page] = 0
        page_distribution[page] += 1
    
    print(f"\nPage distribution: {dict(sorted(page_distribution.items()))}")
    
    # Show some chunks from later pages for verification
    print(f"\nLater chunks for verification:")
    for chunk in chunks[-5:]:  # Last 5 chunks
        start_pos = chunk.get('_debug_start', 'unknown')
        end_pos = chunk.get('_debug_end', 'unknown')
        print(f"Chunk {chunk['chunk_index']} (Page {chunk['page']}): pos {start_pos}-{end_pos}")
        print(f"  Text: {chunk['text'][:100]}...")
        print()