#!/usr/bin/env python
"""
WebSocket Connection Tester for Autocare

This script establishes a WebSocket connection to the server and logs all messages received.
It can also send test messages to verify bidirectional communication.

Usage:
    python test_websocket.py [--host HOST] [--port PORT]

"""

import asyncio
import websockets
import json
import argparse
import logging
import signal
import sys
from datetime import datetime
from app.utils.logger import get_logger_instance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = get_logger_instance("websocket-tester").logger

# Flag to control the main loop
running = True

def handle_signal(signum, frame):
    """Handle interrupt signals to gracefully exit the WebSocket connection."""
    global running
    logger.info(f"Received signal {signum}, closing connection...")
    running = False

async def test_websocket_connection(host, port):
    """
    Connect to the WebSocket server and handle messages.
    
    Args:
        host: The host address of the WebSocket server
        port: The port number of the WebSocket server
    """
    global running
    uri = f"ws://{host}:{port}/ws"
    logger.info(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"Successfully connected to {uri}")
            
            # Send an initial message
            test_message = {"type": "test", "data": "Hello from WebSocket tester", "timestamp": datetime.now().isoformat()}
            await websocket.send(json.dumps(test_message))
            logger.info(f"Sent test message: {test_message}")
            
            # Set up an async task to send a heartbeat message every 30 seconds
            heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
            
            # Listen for messages
            while running:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    try:
                        parsed_message = json.loads(message)
                        logger.info(f"Received message: {json.dumps(parsed_message, indent=2)}")
                    except json.JSONDecodeError:
                        logger.error(f"Received non-JSON message: {message}")
                except asyncio.TimeoutError:
                    # This is expected, it just allows us to check the running flag periodically
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.error("WebSocket connection closed unexpectedly")
                    break
            
            # Cancel the heartbeat task when we're done
            heartbeat_task.cancel()
            
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {str(e)}")

async def send_heartbeat(websocket):
    """Send a heartbeat message every 30 seconds to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(30)
            heartbeat = {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
            await websocket.send(json.dumps(heartbeat))
            logger.info("Sent heartbeat message")
    except asyncio.CancelledError:
        # Task was cancelled, which is expected when shutting down
        pass
    except Exception as e:
        logger.error(f"Error in heartbeat: {str(e)}")

def main():
    """Parse command line arguments and start the WebSocket test."""
    parser = argparse.ArgumentParser(description="Test WebSocket connection to the YouTube Data Scraper server")
    parser.add_argument("--host", default="localhost", help="WebSocket server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port (default: 8000)")
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Start the async event loop
    asyncio.run(test_websocket_connection(args.host, args.port))
    
    logger.info("WebSocket test completed")

if __name__ == "__main__":
    main()
