# CrossEncoder Pipeline with singleton pattern for document reranking
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mxbai_rerank import MxbaiRerankV2

class CrossEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{device}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(CrossEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
            
        self.model_name = model_name
        # Auto-detect best available device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  
        else:
            self.device = "cpu"
        
        if "mxbai" in model_name:
            # MxbaiRerankV2 crashes on MPS, so use CPU for MPS but allow CUDA
            mxbai_device = "cpu" if self.device == "mps" else self.device
            self.model = MxbaiRerankV2(model_name, device=mxbai_device)
            self.model_type = "mxbai"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model_type = "cross_encoder"
        
        self.initialized = True
    
    def rerank(self, query, documents, top_n=4, hybrid_rerank=False, bm25_weight=0.1):
        """Rerank documents using cross-encoder"""
        
        if self.model_type == "mxbai":
            # Get texts out of documents dicts
            texts = [doc["text"] for doc in documents]
            
            # rerank in halves to avoid memory issues
            rerankings_half_1 = self.model.rank(query, texts[:25], top_k=top_n)  # Rerank the chunks  
            rerankings_half_2 = self.model.rank(query, texts[25:], top_k=top_n)  # Rerank the chunks
            for d in rerankings_half_2:
                d.index += 25
            
            # combine rerankings
            rerankings = rerankings_half_1 + rerankings_half_2

            # sort rerankings by score and keep only top_n
            rerankings = sorted(rerankings, key=lambda x: x.score, reverse=True)[:top_n]
            
            reranked_scores = [d.score for d in rerankings]
            reranked_indices = [d.index for d in rerankings]
            reranked_results = []
            for i in range(min(top_n, len(reranked_indices))):
                doc = documents[reranked_indices[i]]
                result = {
                    **{k: v for k, v in doc.items() if k not in ["vector", "match_types"]},
                    "rerank_score": reranked_scores[i]
                }
                
                if hybrid_rerank:
                    result["rerank_score_bm25"] = result["rerank_score"] + float(doc.get("bm25_score", 0)) * bm25_weight
                else:
                    result["rerank_score_bm25"] = result["rerank_score"]
                
                reranked_results.append(result)
            
            # Sort by hybrid score if hybrid reranking is enabled
            if hybrid_rerank:
                reranked_results = sorted(reranked_results, key=lambda x: x['rerank_score_bm25'], reverse=True)[:top_n]
            
            return reranked_results
        else:
            # Original cross-encoder logic
            texts = [doc["text"] for doc in documents]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                [query] * len(texts), # Repeat query for each document
                text_pair=texts, # Pair query with each document (i.e., query + chunk 1, query + chunk 2, ...)
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert logits to scores
            scores = torch.sigmoid(logits).squeeze().cpu().numpy() # Convert to numpy array
            
            # Create results list
            results = []
            for idx, doc in enumerate(documents):
                result = {
                    **doc,
                    "rerank_score": float(scores[idx])
                }
                
                if hybrid_rerank:
                    result["rerank_score_bm25"] = result["rerank_score"] + float(doc.get("bm25_score", 0)) * bm25_weight
                else:
                    result["rerank_score_bm25"] = result["rerank_score"]
                
                results.append(result)
            
            results_sorted = sorted(results, key=lambda x: x['rerank_score_bm25'], reverse=True)
            return results_sorted[:top_n]