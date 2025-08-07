# Progress tracking system for PDF processing and query operations
import time
from datetime import datetime

class BaseProgressTracker:
    """Base class for all progress tracking functionality"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all progress tracking - to be implemented by subclasses"""
        self.current_stage = "idle"
        self.stage_progress = 0
        self.stage_total = 0
        self.overall_progress = 0
        self.is_active = False
        self.detailed_log = []
        self.session_start_time = time.time()
        self.current_detail = ""
    
    def log_message(self, message):
        """Add a timestamped message to the detailed log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.detailed_log.append(log_entry)
        print(log_entry)  # Also print to console/log file
    
    def get_progress_info(self):
        """Get current progress information for UI - to be implemented by subclasses"""
        return {
            "progress": self.overall_progress,
            "stage": self.current_stage,
            "detail": self.current_detail,
            "is_active": self.is_active
        }
    
    def complete(self):
        """Mark processing as complete"""
        self.overall_progress = 100
        self.is_active = False
        self.log_message("Processing complete!")

class ProgressTracker(BaseProgressTracker):
    """Centralized progress tracking for PDF processing"""
    
    def reset(self):
        """Reset all progress tracking"""
        super().reset()
        self.total_files = 0
        self.current_file = ""
        self.current_file_index = 0
        self.stage_weights = {
            "upload": 5,
            "pdf_processing": 40,
            "embedding_init": 10,
            "embedding_creation": 45
        }
        self.stage_messages = {
            "upload": "Uploading files...",
            "pdf_processing": "Converting PDFs to text...",
            "embedding_init": "Initializing AI models...",
            "embedding_creation": "Creating document embeddings..."
        }
    
    def set_stage(self, stage, total_items=1):
        """Set the current processing stage"""
        self.current_stage = stage
        self.stage_total = total_items
        self.stage_progress = 0
        self.is_active = True
        self.log_message(f"Stage: {self.stage_messages.get(stage, stage)}")
    
    def update_stage_progress(self, progress, current_item=""):
        """Update progress within current stage"""
        self.stage_progress = progress
        self.current_file = current_item
        if current_item:
            self.log_message(f"Processing: {current_item}")
        self.calculate_overall_progress()
    
    def increment_stage_progress(self, item_name=""):
        """Increment stage progress by 1"""
        self.stage_progress += 1
        self.current_file = item_name
        if item_name:
            self.log_message(f"Processing: {item_name}")
        self.calculate_overall_progress()
    
    def calculate_overall_progress(self):
        """Calculate overall progress percentage"""
        if self.current_stage == "idle":
            self.overall_progress = 0
            return
        
        # Calculate progress for completed stages
        completed_weight = 0
        stage_order = ["upload", "pdf_processing", "embedding_init", "embedding_creation"]
        
        try:
            current_stage_index = stage_order.index(self.current_stage)
            for i in range(current_stage_index):
                completed_weight += self.stage_weights[stage_order[i]]
        except ValueError:
            current_stage_index = 0
        
        # Calculate progress for current stage
        if self.stage_total > 0:
            current_stage_progress = (self.stage_progress / self.stage_total) * self.stage_weights[self.current_stage]
        else:
            current_stage_progress = 0
        
        self.overall_progress = completed_weight + current_stage_progress
        self.overall_progress = min(100, max(0, self.overall_progress))
    
    def get_progress_info(self):
        """Get current progress information for UI"""
        if self.current_stage == "idle":
            return {
                "progress": 0,
                "stage": "Ready",
                "current_file": "",
                "detailed_message": "Ready to process files"
            }
        
        if self.current_stage == "complete":
            return {
                "progress": 100,
                "stage": "Complete",
                "current_file": "",
                "detailed_message": "All documents processed successfully! Ready to chat."
            }
        
        stage_message = self.stage_messages.get(self.current_stage, self.current_stage)
        
        if self.current_file:
            if self.current_stage == "pdf_processing":
                detailed_message = f"{stage_message} - Converting document {self.stage_progress}/{self.stage_total}: {self.current_file}"
            elif self.current_stage == "embedding_creation":
                detailed_message = f"{stage_message} - Embedding document {self.stage_progress}/{self.stage_total}: {self.current_file}"
            else:
                detailed_message = f"{stage_message} - {self.current_file}"
        else:
            detailed_message = stage_message
        
        return {
            "progress": self.overall_progress,
            "stage": stage_message,
            "current_file": self.current_file,
            "detailed_message": detailed_message
        }
    
    def complete(self):
        """Mark processing as complete"""
        self.current_stage = "complete"
        self.overall_progress = 100
        self.current_file = ""  # Clear current file to avoid confusion
        self.log_message("Processing complete! Ready to chat.")

class CheckProgressTracker(BaseProgressTracker):
    """Progress tracker specifically for check processed PDFs operation"""
    
    def reset(self):
        """Reset all progress tracking"""
        super().reset()
        self.current_step = 0
        self.total_steps = 6  # Based on the log steps
        self.current_status = "Ready to check for processed PDFs"
        self.progress_percentage = 0
    
    def set_step(self, step_num, status, detail=""):
        """Set current step and status"""
        self.current_step = step_num
        self.current_status = status
        self.current_detail = detail
        self.progress_percentage = (step_num / self.total_steps) * 100
        self.is_active = True
    
    def complete(self):
        """Mark as complete"""
        self.current_step = self.total_steps
        self.progress_percentage = 100
        self.current_status = "Check completed successfully"
        self.current_detail = ""
        self.is_active = False
    
    def get_progress_info(self):
        """Get current progress information"""
        return {
            "progress": self.progress_percentage,
            "status": self.current_status,
            "detail": self.current_detail,
            "is_active": self.is_active
        }

class QueryProgressTracker(BaseProgressTracker):
    """Progress tracker for query processing stages"""
    
    def reset(self):
        """Reset all progress tracking"""
        super().reset()
        self.stage_weights = {
            "initializing": 5,
            "retrieving": 30,
            "reranking": 25,
            "generating": 40
        }
        self.stage_messages = {
            "initializing": "Initializing query processing...",
            "retrieving": "Retrieving relevant documents...",
            "reranking": "Reranking retrieved documents...",
            "generating": "Generating response..."
        }
    
    def set_stage(self, stage, total_items=1, detail=""):
        """Set the current processing stage"""
        self.current_stage = stage
        self.stage_total = total_items
        self.stage_progress = 0
        self.current_detail = detail
        self.is_active = True
        message = self.stage_messages.get(stage, stage)
        if detail:
            message += f" - {detail}"
        self.log_message(message)
        self.calculate_overall_progress()
    
    def update_stage_progress(self, progress, detail=""):
        """Update progress within current stage"""
        self.stage_progress = progress
        if detail:
            self.current_detail = detail
            self.log_message(f"Progress: {detail}")
        self.calculate_overall_progress()
    
    def increment_stage_progress(self, detail=""):
        """Increment stage progress by 1"""
        self.stage_progress += 1
        if detail:
            self.current_detail = detail
            self.log_message(f"Progress: {detail}")
        self.calculate_overall_progress()
    
    def calculate_overall_progress(self):
        """Calculate overall progress percentage"""
        if self.current_stage == "idle":
            self.overall_progress = 0
            return
        
        # Calculate progress for completed stages
        completed_weight = 0
        stage_order = ["initializing", "retrieving", "reranking", "generating"]
        
        try:
            current_stage_index = stage_order.index(self.current_stage)
            for i in range(current_stage_index):
                completed_weight += self.stage_weights[stage_order[i]]
        except ValueError:
            current_stage_index = 0
        
        # Calculate progress for current stage
        if self.stage_total > 0:
            current_stage_progress = (self.stage_progress / self.stage_total) * self.stage_weights[self.current_stage]
        else:
            current_stage_progress = 0
        
        self.overall_progress = completed_weight + current_stage_progress
        self.overall_progress = min(100, max(0, self.overall_progress))
    
    def get_progress_info(self):
        """Get current progress information for UI"""
        if self.current_stage == "idle":
            return {
                "progress": 0,
                "stage": "Ready",
                "detail": "",
                "is_active": False,
                "detailed_message": "Ready to process query"
            }
        
        if self.current_stage == "complete":
            return {
                "progress": 100,
                "stage": "Complete",
                "detail": "",
                "is_active": False,
                "detailed_message": "Response generated successfully!"
            }
        
        stage_message = self.stage_messages.get(self.current_stage, self.current_stage)
        
        # Build detailed message
        if self.current_detail:
            detailed_message = f"{stage_message} - {self.current_detail}"
        else:
            detailed_message = stage_message
        
        # Add progress info for stages with multiple items
        if self.stage_total > 1:
            detailed_message += f" ({self.stage_progress}/{self.stage_total})"
        
        return {
            "progress": self.overall_progress,
            "stage": stage_message,
            "detail": self.current_detail,
            "is_active": self.is_active,
            "detailed_message": detailed_message
        }
    
    def complete(self):
        """Mark processing as complete"""
        self.current_stage = "complete"
        self.overall_progress = 100
        self.current_detail = ""
        self.is_active = False
        self.log_message("Query processing complete!")

# Global progress tracker instances
progress_tracker = ProgressTracker()
check_progress_tracker = CheckProgressTracker()
query_progress_tracker = QueryProgressTracker()