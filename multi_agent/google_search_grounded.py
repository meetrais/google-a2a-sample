# --- In your remote agent's Flask app file ---

from flask import Flask, request, jsonify
# ... other imports ...
import logging
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
from dotenv import load_dotenv

# --- CHANGE TO ABSOLUTE IMPORT ---
from common_types import Task, TaskStatus, Message, TextPart, TaskState
# --- END CHANGE ---


load_dotenv()

Gemini_API_Key = os.getenv("Gemini_API_Key")
client = genai.Client(api_key=Gemini_API_Key)

# ... HOST, PORT, BASE_URL definitions ...
HOST = '127.0.0.1'
PORT = 5000
BASE_URL = f"http://{HOST}:{PORT}" # Make sure this is defined

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/agent-card", methods=["GET"])
def agent_card():
    """Return metadata about this agent in JSON format (Agent Card)."""
    logging.info("Received request for Agent Card /agent-card")
    agent_metadata = {
        "name": "GoogleSearchGrounded",
        "description": "Capable of returning results by performing google search.",
        # --- ADD THIS LINE ---
        "url": BASE_URL, # The base URL where this agent is reachable
        # --- END OF ADDITION ---
        "version": "1.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False
        },
        "skills": []
        # Ensure all other fields required by your AgentCard model are present
    }
    return jsonify(agent_metadata)

# --- Keep the rest of your code, including the /tasks endpoint ---
@app.route("/tasks", methods=["POST"])
def handle_task():
    """Handles incoming tasks sent via POST request."""
    print(f"\nReceived POST request to /tasks/send")
    try:
        task = request.get_json()
        if not task:
            return jsonify({"error": "Invalid JSON payload"}), 400

        task_id = task.get("id")
        if not task_id:
             return jsonify({"error": "Missing task ID"}), 400

        system_prompt="Perform Google search and return grounded response to user's question"
        user_message = system_prompt + task.get("message", {}).get("parts", [{}])[0].get("text", "")

        google_search_tool = Tool(
            google_search = GoogleSearch()
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=GenerateContentConfig(
                tools=[google_search_tool], response_modalities=["TEXT"]
            ),
        )
        # Extract the reply text from the response
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            reply_text = response.candidates[0].content.parts[0].text
        else:
            reply_text = "No results found."

        # --- USE PYDANTIC MODELS ---
        # Construct the agent's reply message using models
        agent_reply_message = Message(
            role="agent",
            parts=[TextPart(text=reply_text)] # Use TextPart model
        )

        # Construct the TaskStatus using models
        task_status = TaskStatus(
            state=TaskState.COMPLETED, # Use TaskState enum
            message=agent_reply_message
        )

        # Construct the Task object structure using models
        task_result_payload = Task(
             id=task_id,
             sessionId=task.get("sessionId"), # Pass session ID back if received
             status=task_status,
             # artifacts=[], # Optional: Initialize if needed
             # history=[], # Optional: Initialize if needed
             # metadata={} # Optional: Initialize if needed
        )

        # Construct the full JSONRPCResponse structure
        # Pydantic models will be automatically serialized to JSON by jsonify
        response_payload = {
            "jsonrpc": "2.0",
            "id": task.get("metadata", {}).get("jsonrpc_id", task_id), # Try to reuse JSON-RPC ID if sent, else task_id
            "result": task_result_payload.model_dump(exclude_none=True) # Serialize the Task model
        }
        # --- END USE PYDANTIC MODELS ---

        print(f"  Sending Response Payload: {response_payload}")
        return jsonify(response_payload), 200  # 200 OK

    except (KeyError, IndexError, TypeError) as e:
        print(f"  Error processing request: {e}")
        return jsonify({"error": "Invalid task format"}), 400
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print(f"--- A2A Tell Time Server starting on http://{HOST}:{PORT} ---")
    app.run(host=HOST, port=PORT, debug=False)
