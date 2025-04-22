import sys
import asyncio
import functools
import json
import uuid
import threading
from typing import List, Optional, Callable

from google.genai import types
import base64

from google.adk import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from .remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback
)
from .card_resolver import A2ACardResolver
from .common_types import (
    AgentCard,
    Message,
    TaskState,
    Task,
    TaskSendParams,
    TextPart,
    DataPart,
    Part,
    TaskStatusUpdateEvent,
)

class HostAgent:
  """The host agent.

  This is the agent responsible for choosing which remote agents to send
  tasks to and coordinate their work.
  """

  def __init__(
      self,
      remote_agent_addresses: List[str],
      task_callback: TaskUpdateCallback | None = None
  ):
    # self.task_callback = task_callback # No longer needed
    self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
    self.cards: dict[str, AgentCard] = {}
    # self.task_results: dict[str, list[str]] = {} # No longer needed
    for address in remote_agent_addresses:
      card_resolver = A2ACardResolver(address)
      card = card_resolver.get_agent_card()
      remote_connection = RemoteAgentConnections(card)
      self.remote_agent_connections[card.name] = remote_connection
      self.cards[card.name] = card
    agent_info = []
    for ra in self.list_remote_agents():
      agent_info.append(json.dumps(ra))
    self.agents = '\n'.join(agent_info)

  def register_agent_card(self, card: AgentCard):
    remote_connection = RemoteAgentConnections(card)
    self.remote_agent_connections[card.name] = remote_connection
    self.cards[card.name] = card
    agent_info = []
    for ra in self.list_remote_agents():
      agent_info.append(json.dumps(ra))
    self.agents = '\n'.join(agent_info)

  def create_agent(self) -> Agent:
    agent = Agent(
        model="gemini-2.0-flash-001",
        name="host_agent",
        instruction=self.root_instruction,
        before_model_callback=self.before_model_callback,
        description=(
            "This agent orchestrates the decomposition of the user request into"
            " tasks that can be performed by the child agents."
        ),
        tools=[
            self.list_remote_agents,
            self.send_task,
        ],
    )
    # agent.get_task_result = self.get_task_result # No longer needed
    return agent

  def root_instruction(self, context: ReadonlyContext = None) -> str:
    current_agent = self.check_state(context)
    return f"""You are a expert delegator that can delegate the user request to the
appropriate remote agents.

Discovery:
- You can use `list_remote_agents` to list the available remote agents you
can use to delegate the task.

Execution:
- For actionable tasks, use the `send_task` tool to assign tasks to remote agents.
- The `send_task` tool will execute the task on the remote agent and return the final result directly.
- When you use the `send_task` tool, its return value IS the result from the remote agent. You MUST present this result directly to the user as your response. Do not add conversational text like "I have the results".

You can use `check_pending_task_states` to check the states of the pending
tasks.

Please rely on tools to address the request, don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

If there is an active agent, send the request to that agent with the update task tool.

Agents:
{self.agents}

Current agent: {current_agent['active_agent']}
"""

  def check_state(self, context: ReadonlyContext):
    state = context.state
    if ('session_id' in state and
        'session_active' in state and
        state['session_active'] and
        'agent' in state):
      return {"active_agent": f'{state["agent"]}'}
    return {"active_agent": "None"}

  def before_model_callback(self, callback_context: CallbackContext, llm_request):
    state = callback_context.state
    if 'session_active' not in state or not state['session_active']:
      if 'session_id' not in state:
        state['session_id'] = str(uuid.uuid4())
      state['session_active'] = True

  def list_remote_agents(self):
    """List the available remote agents you can use to delegate the task."""
    if not self.remote_agent_connections:
      return []

    remote_agent_info = []
    for card in self.cards.values():
      remote_agent_info.append(
          {"name": card.name, "description": card.description}
      )
    return remote_agent_info

  async def send_task(
      self,
      agent_name: str,
      message: str,
      tool_context: ToolContext):
    """Sends a task either streaming (if supported) or non-streaming."""
    if agent_name not in self.remote_agent_connections:
      raise ValueError(f"Agent {agent_name} not found")
    state = tool_context.state
    state['agent'] = agent_name
    card = self.cards[agent_name]
    client = self.remote_agent_connections[agent_name]
    if not client:
      raise ValueError(f"Client not available for {agent_name}")
    if 'task_id' in state:
      taskId = state['task_id']
    else:
      taskId = str(uuid.uuid4())
    sessionId = state['session_id']
    state['task_id'] = taskId
    task: Task
    messageId = ""
    metadata = {}
    if 'input_message_metadata' in state:
      metadata.update(**state['input_message_metadata'])
      if 'message_id' in state['input_message_metadata']:
        messageId = state['input_message_metadata']['message_id']
    if not messageId:
      messageId = str(uuid.uuid4())
    metadata.update(**{'conversation_id': sessionId, 'message_id': messageId})
    request: TaskSendParams = TaskSendParams(
        id=taskId,
        sessionId=sessionId,
        message=Message(
            role="user",
            parts=[TextPart(text=message)],
            metadata=metadata,
        ),
        acceptedOutputModes=["text", "text/plain", "image/png"],
        # pushNotification=None,
        metadata={'conversation_id': sessionId},
    )
    # For non-streaming, send_task returns the final task object
    task = await client.send_task(request, None) # Pass None for callback

    response = []
    if task and task.status and task.status.message:
      # Extract results directly from the final task object
      response.extend(self.convert_parts(task.status.message.parts, tool_context))
    elif task and task.status:
       response.append(f"Task completed with status: {task.status.state}")
    else:
       response.append(f"Task sent to {agent_name}, but no response message received.")

    if task and task.artifacts:
      for artifact in task.artifacts:
        response.extend(self.convert_parts(artifact.parts, tool_context))

    # Join the list into a single string response
    return "\n".join(map(str, response)) if response else "No response content received from agent."

  # Removed get_task_result method
  # Removed task_callback method

  def convert_parts(self, parts: list[Part], tool_context: ToolContext):
    """Converts a list of parts to a list of strings."""
    rval = []
    for p in parts:
      rval.append(self.convert_part(p, tool_context))
    return rval

  def convert_part(self, part: Part, tool_context: ToolContext):
    """Converts a part to a string."""
    if part.type == "text":
      return part.text
    elif part.type == "data":
      return part.data
    elif part.type == "file":
      # Repackage A2A FilePart to google.genai Blob
      # Currently not considering plain text as files
      file_id = part.file.name
      file_bytes = base64.b64decode(part.file.bytes)
      file_part = types.Part(
          inline_data=types.Blob(mime_type=part.file.mimeType, data=file_bytes)
      )
      if tool_context:
        tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
      return DataPart(data={"artifact-file-id": file_id})
    return f"Unknown type: {part.type}"
