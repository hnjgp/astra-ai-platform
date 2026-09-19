from collections.abc import Collection

from tools.registry import TOOL_REGISTRY


class ToolAuthorization:
    def __init__(
        self,
        role_permissions: dict[str, Collection[str]]
        | None = None,
    ):
        self.role_permissions = (
            {
                role: set(tools)
                for role, tools in role_permissions.items()
            }
            if role_permissions is not None
            else {
                "admin": set(TOOL_REGISTRY),
                "user": set(TOOL_REGISTRY),
                "guest": set(TOOL_REGISTRY),
            }
        )

    def check(
        self,
        role: str,
        tool_name: str,
    ) -> None:
        if not role.strip():
            raise ValueError(
                "role must not be empty"
            )

        if role not in self.role_permissions:
            raise ValueError(
                f"Unknown role: {role}"
            )

        if tool_name not in self.role_permissions[role]:
            raise ValueError(
                f"Tool is not allowed for role: "
                f"{role}"
            )