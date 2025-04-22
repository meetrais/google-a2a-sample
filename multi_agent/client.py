# multi_agent/client.py

import httpx
import logging
from .common_types import ( # Ensure all needed types are imported
    A2AClientHTTPError,
    A2AClientConnectionError, # <--- CHECK SPELLING HERE
    SendTaskResponse,
    AgentCard # If get_agent_card is implemented here
)

# Setup logger for this module
logger = logging.getLogger(__name__)
# Configure logging level if needed (e.g., in your main script or agent.py)
# logging.basicConfig(level=logging.INFO) # Example basic config

class A2AClient:
    """Client for interacting with a remote A2A agent."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initializes the A2AClient.

        Args:
            base_url: The base URL of the remote agent (e.g., http://127.0.0.1:5000).
            timeout: Default timeout in seconds for requests.
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        self.base_url = base_url.rstrip('/') # Remove trailing slash if present
        self.timeout_config = httpx.Timeout(timeout, connect=5.0)
        logger.info(f"A2AClient initialized for base URL: {self.base_url}")

    async def _send_request(self, method: str, path: str, json_data: dict | None = None) -> dict:
        """Sends an HTTP request to the specified path on the remote agent.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            path: The API path (e.g., "/agent-card", "/tasks"). It MUST start with '/'.
            json_data: Optional dictionary payload for POST/PUT requests.

        Returns:
            The JSON response dictionary from the agent.

        Raises:
            A2AClientHTTPError: If the agent returns a 4xx or 5xx status.
            A2AClientConnectionError: If a connection-level error occurs.
            ValueError: If the path is invalid or method is unsupported.
            Exception: For other unexpected errors (like JSON parsing).
        """
        if not path or not path.startswith('/'):
             logger.error(f"Invalid path provided to _send_request: '{path}'")
             raise ValueError("API path must start with '/'")

        # --- CRUCIAL: Construct the full URL ---
        url = f"{self.base_url}{path}"
        # --- END CRUCIAL PART ---

        # Log the exact URL being requested BEFORE making the call
        logger.info(f"Sending A2A request: {method.upper()} {url}")
        if json_data:
            logger.debug(f"Request payload: {json_data}") # Be careful logging sensitive data

        # Use a context manager for the client to ensure proper cleanup
        async with httpx.AsyncClient(timeout=self.timeout_config) as client:
            try:
                if method.upper() == "POST":
                    if json_data is None:
                         logger.warning(f"POST request to {url} called without JSON data.")
                         # Decide if this is an error or if empty body is allowed
                         # raise ValueError("POST request requires json_data")
                    response = await client.post(url, json=json_data)
                elif method.upper() == "GET":
                    response = await client.get(url)
                else:
                    logger.error(f"Unsupported HTTP method requested: {method}")
                    raise ValueError(f"Unsupported HTTP method: {method}")

                logger.debug(f"A2A Response Status for {method.upper()} {url}: {response.status_code}")

                # Check for HTTP errors AFTER getting the response
                response.raise_for_status()

                # Handle successful responses
                if response.status_code == 204: # No Content
                    return {}
                # Ensure response has content before trying to parse JSON
                if not response.content:
                    logger.warning(f"Received empty response body for {method.upper()} {url} (Status: {response.status_code})")
                    return {} # Or raise an error if JSON is always expected

                return response.json()

            except httpx.HTTPStatusError as e:
                 # Log the specific error before re-raising
                 logger.error(f"HTTP error {e.response.status_code} for URL {e.request.url}: {e}", exc_info=True)
                 raise A2AClientHTTPError(e.response.status_code, str(e)) from e
            except httpx.RequestError as e:
                 # Log connection errors
                 logger.error(f"Request error for URL {e.request.url}: {e}", exc_info=True)
                 raise A2AClientConnectionError(str(e)) from e
            except Exception as e:
                 # Catch other potential errors (e.g., JSONDecodeError)
                 logger.exception(f"Unexpected error during request to {url}: {e}")
                 # Re-raise the original exception or a generic one
                 raise

    async def send_task(self, request_data: dict) -> SendTaskResponse:
        """Sends a task creation request (POST /tasks) to the remote agent.

        Args:
            request_data: The dictionary payload conforming to TaskSendParams.

        Returns:
            A SendTaskResponse object parsed from the agent's response.

        Raises:
            TypeError: If request_data is not a dictionary.
            ValueError: If the response format from the agent is invalid.
            A2AClientHTTPError: For HTTP errors from the agent.
            A2AClientConnectionError: For connection errors.
        """
        # --- CRUCIAL: Define the correct path ---
        task_path = "/tasks"
        # --- END CRUCIAL PART ---

        logger.debug(f"Calling _send_request for send_task with path: {task_path}")

        if not isinstance(request_data, dict):
             logger.error(f"Invalid request_data type for send_task: {type(request_data)}")
             raise TypeError("request_data for send_task must be a dictionary")

        # --- CRUCIAL: Call _send_request with the correct path ---
        logger.info(f"Sending task request: {request_data}")
        response_json = await self._send_request(method="POST", path=task_path, json_data=request_data)
        # --- END CRUCIAL PART ---

        # Validate and parse the response
        try:
            # Ensure SendTaskResponse is imported and correctly defined in common_types.py
            task_response = SendTaskResponse(**response_json)
            logger.info(f"Successfully received and parsed response for task {task_response.id}")
            return task_response
        except Exception as e: # Catch Pydantic validation errors or others
            logger.exception(f"Failed to parse SendTaskResponse from {task_path}. Response JSON: {response_json}")
            raise ValueError("Invalid response format received from agent for send_task") from e

    # Optional: Keep get_agent_card if this client is also used for that
    async def get_agent_card(self) -> AgentCard:
         """Fetches the agent card (GET /agent-card) from the remote agent."""
         card_path = "/agent-card"
         logger.debug(f"Calling _send_request for get_agent_card with path: {card_path}")
         response_json = await self._send_request(method="GET", path=card_path)
         try:
             # Ensure AgentCard is imported and correctly defined
             agent_card = AgentCard(**response_json)
             logger.info(f"Successfully received and parsed agent card for {agent_card.name}")
             return agent_card
         except Exception as e:
             logger.exception(f"Failed to parse AgentCard from {card_path}. Response JSON: {response_json}")
             raise ValueError("Invalid response format received from agent for get_agent_card") from e
