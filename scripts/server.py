#!/usr/bin/env python
"""
server.py

Model server that keeps the model loaded in memory for fast inference.
Run this once, then use client.py for fast text generation.
"""

import sys
import os
import json
import socket
import threading
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.text_generator import TextGenerator
from src.utils import setup_environment

class ModelServer:
    def __init__(self, config_path="config/generation_config.json", port=8765):
        self.port = port
        self.generator = None
        self.running = False
        
        # Load config and initialize generator
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("🚀 Starting model server...")
        setup_environment(verbose=False)
        self.generator = TextGenerator(config)
        print(f"✅ Model server ready on port {port}")
        
    def handle_client(self, client_socket):
        """Handle client request"""
        try:
            # Receive prompt
            prompt = client_socket.recv(4096).decode('utf-8')
            if not prompt:
                return
                
            print(f"📝 Generating response for: {prompt[:50]}...")
            
            # Generate response
            response = self.generator.generate(prompt)
            
            # Send response
            client_socket.send(response.encode('utf-8'))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            client_socket.send(error_msg.encode('utf-8'))
            
        finally:
            client_socket.close()
    
    def start(self):
        """Start the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen(5)
        
        self.running = True
        print(f"🌐 Model server listening on localhost:{self.port}")
        print("💡 Use 'python client.py \"prompt\"' for fast generation")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                client_socket, addr = server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket,)
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            
        finally:
            server_socket.close()

if __name__ == "__main__":
    server = ModelServer()
    server.start()