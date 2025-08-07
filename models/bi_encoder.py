# BiEncoder Pipeline with singleton pattern for embedding generation
import os
import pickle
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BiEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{chunk_size}_{chunk_overlap}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(BiEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
        
        from models.progress_tracking import progress_tracker
        progress_tracker.log_message(f"Initializing BiEncoderPipeline with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.initialized = True
    
    def embed_page_aware_documents(self, page_chunks_list, doc_ids=None, save_path=None, track_progress=True):
        """Embed documents using page-aware chunks with optional progress tracking"""
        from models.progress_tracking import progress_tracker
        
        # If single document page chunks given, make it a list
        if not isinstance(page_chunks_list, list) or (len(page_chunks_list) > 0 and isinstance(page_chunks_list[0], dict)):
            page_chunks_list = [page_chunks_list]
        
        if save_path is not None:
            actual_save_path = Path(save_path) / self.model_name.split("/")[-1] / f"chunk_size_{self.chunk_size}" / f"chunk_overlap_{self.chunk_overlap}"
            if track_progress:
                progress_tracker.log_message(f"Actual save path: {actual_save_path}")
                # Create save path if it doesn't exist
                progress_tracker.log_message(f"Creating save directory: {actual_save_path}")
            if not os.path.exists(actual_save_path):
                os.makedirs(actual_save_path)
        
        # Process each document's page chunks
        results = []
        
        for i, page_chunks in enumerate(page_chunks_list):
            doc_name = doc_ids[i] if doc_ids else f'doc_{i}'
            if track_progress:
                progress_tracker.log_message(f"Embedding document {i+1}/{len(page_chunks_list)}: {doc_name}")
                progress_tracker.increment_stage_progress(doc_name)
            
            # Create a directory for the document if it doesn't exist
            if save_path and doc_ids:
                doc_dir = actual_save_path / doc_ids[i]
                if not os.path.exists(doc_dir):
                    os.makedirs(doc_dir)
            
            # Check if all chunks for this document are already saved
            if save_path is not None and doc_ids:
                all_chunks_exist = all(
                    os.path.exists(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl") 
                    for j in range(len(page_chunks))
                )
                
                if all_chunks_exist:
                    if track_progress:
                        progress_tracker.log_message(f"Skipping {doc_ids[i]} because all chunks already exist")
                    # Load existing chunks from disk
                    for j in range(len(page_chunks)):
                        with open(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl", "rb") as f:
                            chunk_data = pickle.load(f)
                            results.append(chunk_data)
                    continue
            
            # Extract texts for embedding
            chunk_texts = [chunk['text'] for chunk in page_chunks]
            chunk_vectors = self.model.encode(chunk_texts)
            
            # Create results with page information
            for j, (chunk, vector) in enumerate(zip(page_chunks, chunk_vectors)):
                chunk_result = {
                    "text": chunk['text'],
                    "vector": vector,
                    "page": chunk['page'],
                    "original_doc_id": doc_ids[i] if doc_ids is not None else None,
                    "doc_idx": i,
                    "chunk_idx": j
                }
                results.append(chunk_result)
                
                # Save chunk with page information
                if save_path is not None and doc_ids:
                    chunk_path = f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl"
                    if not os.path.exists(chunk_path):
                        with open(chunk_path, "wb") as f:
                            pickle.dump(chunk_result, f)
        
        return results

    def embed_documents(self, doc_texts, doc_ids=None, save_path=None, track_progress=True):
        """Embed documents using pre-loaded models with optional progress tracking"""
        from models.progress_tracking import progress_tracker
        
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_texts, list):
            doc_texts = [doc_texts]
        
        if save_path is not None:
            actual_save_path = Path(save_path) / self.model_name.split("/")[-1] / f"chunk_size_{self.chunk_size}" / f"chunk_overlap_{self.chunk_overlap}"
            if track_progress:
                progress_tracker.log_message(f"Actual save path: {actual_save_path}")
                # Create save path if it doesn't exist
                progress_tracker.log_message(f"Creating save directory: {actual_save_path}")
            if not os.path.exists(actual_save_path):
                os.makedirs(actual_save_path)

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        
        for i, doc in enumerate(doc_texts):
            doc_name = doc_ids[i] if doc_ids else f'doc_{i}'
            if track_progress:
                progress_tracker.log_message(f"Embedding document {i+1}/{len(doc_texts)}: {doc_name}")
                progress_tracker.increment_stage_progress(doc_name)
            
            # Create a directory for the document if it doesn't exist
            if save_path and doc_ids:
                doc_dir = actual_save_path / doc_ids[i]
                if not os.path.exists(doc_dir):
                    os.makedirs(doc_dir)
            
            # Split the document into chunks to check if all are already saved
            doc_chunks = self.text_splitter.split_text(doc)
            
            # Check if all chunks for this document are already saved
            if save_path is not None and doc_ids:
                all_chunks_exist = all(
                    os.path.exists(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl") 
                    for j in range(len(doc_chunks))
                )
                
                if all_chunks_exist:
                    if track_progress:
                        progress_tracker.log_message(f"Skipping {doc_ids[i]} because all chunks already exist")
                    # Load existing chunks from disk
                    loaded_chunks = []
                    loaded_vectors = []
                    for j in range(len(doc_chunks)):
                        with open(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl", "rb") as f:
                            chunk_data = pickle.load(f)
                            loaded_chunks.append(chunk_data["text"])
                            loaded_vectors.append(chunk_data["vector"])
                    
                    all_chunks.append(loaded_chunks)
                    all_vectors.append(loaded_vectors)
                    continue
            
            # Store the chunks and their embeddings
            all_chunks.append(doc_chunks)
            all_vectors.append(self.model.encode(doc_chunks))

        # Create results list; each element is a dict with the chunk text, its vector, its index, and the overall document index
        results = []
        for i, doc_chunks in enumerate(all_chunks): # For each document
            for j, chunk in enumerate(doc_chunks): # For each chunk
                results.append({
                    "text": chunk,
                    "vector": all_vectors[i][j],
                    "original_doc_id": doc_ids[i] if doc_ids is not None else None,
                    "doc_idx": i,
                    "chunk_idx": j
                })
            
                if save_path is not None and doc_ids:
                    # Only save if the chunk doesn't already exist
                    chunk_path = f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl"
                    if not os.path.exists(chunk_path):
                        with open(chunk_path, "wb") as f:
                            pickle.dump(results[-1], f)

        return results
    
    def retrieve_top_k(self, query, documents_embeddings, top_k=50):
        """Retrieve top k embeddings using cosine similarity to the query"""
        
        # Embed query
        query_vector = self.model.encode([query])   
        
        # Get embeddings from documents dicts
        stored_vectors = np.array([item["vector"] for item in documents_embeddings])
        
        # Compute similarities between query and stored vectors
        similarities = cosine_similarity(query_vector, stored_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                **documents_embeddings[i],
                "similarity": float(similarities[i])
            }
            for i in top_indices
        ]
    
    def retrieve_all(self, query, documents_embeddings):
        """Retrieve all embeddings using cosine similarity to the query"""
        
        # Embed query
        query_vector = self.model.encode([query])   
        
        # Get embeddings from documents dicts
        stored_vectors = np.array([item["vector"] for item in documents_embeddings])
        
        # Compute similarities between query and stored vectors
        similarities = cosine_similarity(query_vector, stored_vectors).flatten()
        
        return [
            {
                **documents_embeddings[i],
                "similarity": float(similarities[i])
            }
            for i in range(len(documents_embeddings))
        ]