import json
import re

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


MAX_PLAN_STEPS = 10

_JSON_FENCE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    """
    Models sometimes wrap JSON in a markdown code fence.
    """

    match = _JSON_FENCE.match(text)

    return match.group(1) if match else text


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
            data = json.loads(
                _strip_code_fence(response)
            )
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Planner returned invalid JSON"
            ) from exc

        try:
            plan = Plan.model_validate(data)
        except ValueError as exc:
            raise LLMError(
                "Planner returned an invalid plan"
            ) from exc

        if len(plan.steps) > MAX_PLAN_STEPS:
            raise LLMError(
                "Planner returned too many steps"
            )

        expected_numbers = list(
            range(1, len(plan.steps) + 1)
        )

        if [
            step.step for step in plan.steps
        ] != expected_numbers:
            raise LLMError(
                "Planner returned an invalid plan"
            )

        return plan