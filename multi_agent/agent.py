# agent.py

import os # Example: If addresses come from environment variables
from .host_agent import HostAgent
from google.adk.agents import Agent
# You might need these if your context types are used elsewhere, but not strictly for this fix
# from google.adk.agents.readonly_context import ReadonlyContext
# from google.adk.agents.callback_context import CallbackContext

# --- Step 1: Create an instance of HostAgent ---
# You need to provide the remote agent addresses here.
# How you get these addresses depends on your application setup.
# Example: Reading from environment variables or a config file.
# For demonstration, using an empty list. Replace with your actual logic.
# remote_addresses = os.getenv("REMOTE_AGENT_ADDRESSES", "").split(",") if os.getenv("REMOTE_AGENT_ADDRESSES") else []
remote_addresses = ["http://127.0.0.1:5000"] # Or ["address1", "address2"] or load dynamically

# Ensure you handle potential errors during HostAgent initialization if needed
try:
    # Create the instance
    host_agent_instance = HostAgent(
        remote_agent_addresses=remote_addresses,
        # task_callback=your_callback_function # Optional: If you have a callback
    )
except Exception as e:
    print(f"Error initializing HostAgent: {e}")
    # Handle the error appropriately, maybe exit or provide a default agent
    raise # Re-raise the exception if initialization is critical

# --- Step 2: Use methods from the instance ---
root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="host_agent", # Consider if this name should be dynamic or match the instance
    # Pass the instance methods
    instruction=host_agent_instance.root_instruction,
    before_model_callback=host_agent_instance.before_model_callback,
    description=(
        "This agent orchestrates the decomposition of the user request into"
        " tasks that can be performed by the child agents."
    ),
    tools=[
        host_agent_instance.list_remote_agents,
        host_agent_instance.send_task,
    ],
)

# You can now use 'root_agent' elsewhere in your application