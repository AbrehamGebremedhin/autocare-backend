from .base import BaseService
import requests
import asyncio

class FetchCarDataService(BaseService):
    """
    Service to download a PDF file from a given URL and optionally notify via websocket.
    """
    def notify_websocket(self, message):
        """
        Sends a websocket notification message.
        Handles both async and sync contexts.
        Args:
            message (str): The message to send via websocket.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.websocket_manager.broadcast(message))
        except RuntimeError:
            asyncio.run(self.websocket_manager.broadcast(message))

    def perform_action(self, url: str, output_path: str) -> None:
        """
        Downloads a PDF file from the given URL and saves it to the specified output path.
        Optionally sends a websocket message on success or failure.
        Args:
            url (str): The URL of the PDF file to download.
            output_path (str): The local file path to save the downloaded PDF.
        Raises:
            Exception: If the download fails or the response is not a PDF.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            self.notify_websocket(f"PDF downloaded successfully: {output_path}")
        else:
            self.notify_websocket(f"Failed to download PDF from {url}")
            raise Exception(f"Failed to download PDF. Status code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")

# Example usage (remove or comment out in production):
FetchCarDataService().perform_action("https://manuals.startmycar.com/published/Toyota-Echo_2001_EN-US_US_9e5148ca82.pdf", "Toyota_Echo_2001_EN-US_US_9e5148ca82.pdf")