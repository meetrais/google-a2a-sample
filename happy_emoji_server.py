from flask import Flask, request, jsonify
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

HOST = '127.0.0.1'
PORT = 5000
BASE_URL = f"http://{HOST}:{PORT}"

app = Flask(__name__)

@app.route("/.well-known/agent.json", methods=["GET"])
def agent_card():
    """Return metadata about this agent in JSON format."""
    print("Received request for Agent Card /.well-known/agent.json")
    agent_metadata = {
        "name": "HappyEmoji",  # Human-readable name of the agent
        "description": "Returns response by adding Happy emojis.", # Short summary of capabilities
        "url": BASE_URL, # Where this agent is hosted (used by client to send tasks)
        "version": "1.0", # Version info for the agent
        "capabilities": {
            "streaming": False, # Indicates that this agent does not support streaming updates
            "pushNotifications": False # Indicates that the agent does not send push notifications
        }
    }
    return jsonify(agent_metadata)

@app.route("/tasks/send", methods=["POST"])
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

        system_prompt="Respond to user queries by adding happy emojis to the response. Add as many happy emojis as you can. "
        user_message = system_prompt + task.get("message", {}).get("parts", [{}])[0].get("text", "")
        
        Gemini_API_Key = os.getenv("Gemini_API_Key")
        client = genai.Client(api_key=Gemini_API_Key)

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=user_message
        )

        reply_text = f"{response}"

        response_payload = {
            "id": task_id,  # Reuse the same task ID in the response
            "status": {"state": "completed"}, # Mark the task as completed
            "messages": [
                task.get("message", {}),
                {
                    "role": "agent", # This message is from the agent
                    "parts": [{"text": reply_text}] # Reply content in text format
                }
            ]
        }
        print(f"  Sending Response Payload: {response_payload}")
        return jsonify(response_payload), 200 # 200 OK

    except (KeyError, IndexError, TypeError) as e:
        print(f"  Error processing request: {e}")
        return jsonify({"error": "Invalid task format"}), 400
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print(f"--- A2A Tell Time Server starting on http://{HOST}:{PORT} ---")
    app.run(host=HOST, port=PORT, debug=False)