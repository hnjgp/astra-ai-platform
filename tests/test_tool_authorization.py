import pytest

from tools.authorization import ToolAuthorization


def test_admin_can_access_registered_tool():

    authorization = ToolAuthorization()

    authorization.check(
        role="admin",
        tool_name="get_system_status",
    )


def test_user_can_access_registered_tool():

    authorization = ToolAuthorization()

    authorization.check(
        role="user",
        tool_name="get_system_status",
    )


def test_guest_can_access_registered_tool():

    authorization = ToolAuthorization()

    authorization.check(
        role="guest",
        tool_name="get_system_status",
    )


def test_unknown_role_is_rejected():

    authorization = ToolAuthorization()

    with pytest.raises(
        ValueError,
        match="Unknown role",
    ):
        authorization.check(
            role="manager",
            tool_name="get_system_status",
        )


def test_empty_role_is_rejected():

    authorization = ToolAuthorization()

    with pytest.raises(
        ValueError,
        match="role must not be empty",
    ):
        authorization.check(
            role="",
            tool_name="get_system_status",
        )


def test_role_specific_tool_permission():

    authorization = ToolAuthorization(
        role_permissions={
            "admin": {
                "get_system_status",
                "delete_database",
            },
            "user": {
                "get_system_status",
            },
        }
    )

    authorization.check(
        role="admin",
        tool_name="delete_database",
    )

    with pytest.raises(
        ValueError,
        match="Tool is not allowed for role",
    ):
        authorization.check(
            role="user",
            tool_name="delete_database",
        )