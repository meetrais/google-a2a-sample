# Inside multi_agent/card_resolver.py (or similar)

import httpx
import json
from .common_types import AgentCard # Assuming AgentCard is defined here or imported

class A2ACardResolver:
    def __init__(self, base_url: str):
        # Ensure base_url doesn't have a trailing slash for clean joining
        self.base_url = base_url.rstrip('/')
        # Consider adding timeout defaults
        self.timeout = httpx.Timeout(10.0, connect=5.0)

    def get_agent_card(self) -> AgentCard:
        # --- MAKE THE CHANGE HERE ---
        # OLD PATH (Likely what you have now):
        # card_path = "/.well-known/agent.json"
        # NEW PATH (Standard A2A):
        card_path = "/agent-card"
        # --- END OF CHANGE ---

        url = f"{self.base_url}{card_path}"
        print(f"CardResolver: Attempting to GET Agent Card from: {url}") # Added print for debugging

        try:
            # Use httpx.Client for synchronous code if not in async context
            # Or httpx.AsyncClient if get_agent_card is async
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                print(f"CardResolver: Received status code {response.status_code} from {url}") # Debug status
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
                card_data = response.json()
                # Validate data against the Pydantic model
                return AgentCard(**card_data)
        except httpx.HTTPStatusError as e:
             print(f"CardResolver: HTTP error fetching agent card from {url}: {e}")
             # Re-raise or handle appropriately
             raise e
        except (httpx.RequestError, json.JSONDecodeError) as e:
             print(f"CardResolver: Error fetching or parsing agent card from {url}: {e}")
             # Re-raise or handle appropriately
             raise ValueError(f"Could not fetch or parse agent card from {url}") from e

    # Add other methods if they exist...