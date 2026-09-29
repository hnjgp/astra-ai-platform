import json

from exceptions import LLMError
from schemas import Plan


PLANNER_INSTRUCTIONS = """
You are the planning component of an AI agent.

Create a concise execution plan for the user's request.

Return only valid JSON.

The JSON must have this structure:

{
  "steps": [
    {
      "step": 1,
      "description": "..."
    }
  ]
}

Rules:
- Use the minimum number of steps needed.
- Each step must describe one meaningful action.
- Step numbers must start at 1.
- Do not include any text outside the JSON object.
"""


class Planner:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def create_plan(
        self,
        message: str,
    ) -> Plan:

        response = self.llm_client.generate(
            message=message,
            instructions=PLANNER_INSTRUCTIONS,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Planner returned invalid JSON"
            ) from exc

        try:
            return Plan.model_validate(data)
        except ValueError as exc:
            raise LLMError(
                "Planner returned an invalid plan"
            ) from exc