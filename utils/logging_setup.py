# Logging configuration and setup utilities
import logging
import sys
from datetime import datetime

LOG_FILE = 'app_log.txt'

class FileLogHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
    
    def emit(self, record):
        with open(self.filename, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {self.format(record)}\n")

class PrintLogger:
    def write(self, message):
        if message.strip():
            logger.info(message.strip())
    
    def flush(self):
        pass

def setup_logging():
    """Set up logging configuration"""
    global logger
    
    # Setup logging to file
    with open(LOG_FILE, 'w') as f:
        f.write('')
    
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = FileLogHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    # Redirect print to logging
    original_stdout = sys.stdout
    sys.stdout = PrintLogger()
    
    return logger, original_stdout

# Initialize logger
logger = None


def log_message(message):
    """Log a message to the application log file"""
    global logger
    if logger is None:
        # Initialize basic logging if not set up yet
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger()
    
    logger.info(message)
    print(message)  # Also print to console