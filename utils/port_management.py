# Port management utilities for multi-instance application support
import atexit
import logging
import os
import socket
from datetime import datetime

class PortManager:
    """Manages port allocation and tracking for multiple app instances"""
    
    def __init__(self, log_file="port_log.txt", start_port=8050, max_port=8100):
        self.log_file = log_file
        self.start_port = start_port
        self.max_port = max_port
        self.current_port = None
        self.pid = os.getpid()
        
    def is_port_available(self, port):
        """Check if a port is available for use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0
        except:
            return False
    
    def read_port_log(self):
        """Read current port allocations from log file"""
        if not os.path.exists(self.log_file):
            return {}
        
        active_ports = {}
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            # Check if process is still running
                            if self.is_process_running(int(pid)):
                                active_ports[int(port)] = {'pid': int(pid), 'timestamp': timestamp}
                        except ValueError:
                            continue
        except:
            pass
        
        return active_ports
    
    def is_process_running(self, pid):
        """Check if a process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def find_available_port(self):
        """Find the next available port"""
        active_ports = self.read_port_log()
        
        for port in range(self.start_port, self.max_port + 1):
            if port not in active_ports and self.is_port_available(port):
                return port
        
        raise RuntimeError(f"No available ports found in range {self.start_port}-{self.max_port}")
    
    def register_port(self, port):
        """Register current port usage in log file"""
        self.current_port = port
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean up stale entries first
        self.cleanup_stale_entries()
        
        # Add current entry
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{self.pid}:{port}:{timestamp}\n")
        except:
            pass
        
        # Register cleanup on exit
        atexit.register(self.cleanup_on_exit)
    
    def cleanup_stale_entries(self):
        """Remove entries for processes that are no longer running"""
        if not os.path.exists(self.log_file):
            return
        
        active_entries = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            if self.is_process_running(int(pid)):
                                active_entries.append(line)
                        except ValueError:
                            continue
            
            # Rewrite file with only active entries
            with open(self.log_file, 'w') as f:
                for entry in active_entries:
                    f.write(entry + '\n')
        except:
            pass
    
    def cleanup_on_exit(self):
        """Clean up current port entry when app exits"""
        if not self.current_port:
            return
        
        try:
            active_entries = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            pid, port, timestamp = line.split(':')
                            if int(pid) != self.pid:
                                active_entries.append(line)
                        except ValueError:
                            active_entries.append(line)
            
            with open(self.log_file, 'w') as f:
                for entry in active_entries:
                    f.write(entry + '\n')
        except:
            pass
    
    def get_allocated_port(self):
        """Get an available port and register it"""
        port = self.find_available_port()
        self.register_port(port)
        return port

def start_app():
    """Start the app with automatic port allocation"""
    # Configure logging to reduce Werkzeug verbosity
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    port_manager = PortManager()
    
    try:
        port = port_manager.get_allocated_port()
        return port
    except RuntimeError as e:
        print(f"Error allocating port: {e}")
        return None