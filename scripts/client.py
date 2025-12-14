#!/usr/bin/env python
"""
client.py

Fast client for text generation using the model server.
Usage: python client.py "Your prompt here"
"""

import sys
import socket
import argparse

def generate_text(prompt, port=8765):
    """Send prompt to server and get response"""
    try:
        # Connect to server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', port))
        
        # Send prompt
        client_socket.send(prompt.encode('utf-8'))
        
        # Receive response
        response = b''
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response += chunk
            
        client_socket.close()
        return response.decode('utf-8')
        
    except ConnectionRefusedError:
        return "Error: Model server not running. Start it with: python server.py"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Fast text generation client")
    parser.add_argument("prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    
    args = parser.parse_args()
    
    # Generate text
    response = generate_text(args.prompt, args.port)
    print(response)

if __name__ == "__main__":
    main()