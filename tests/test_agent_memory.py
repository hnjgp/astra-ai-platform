# tests/test_agent_memory.py

from agents.memory import Memory


def test_memory_starts_empty():

    memory = Memory()

    assert (
        memory.retrieve("user_goal")
        is None
    )


def test_memory_can_save_value():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    assert (
        memory.retrieve("user_goal")
        == "AI Engineer"
    )


def test_memory_can_store_multiple_values():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    memory.save(
        "learning_level",
        "beginner",
    )

    assert (
        memory.retrieve("user_goal")
        == "AI Engineer"
    )

    assert (
        memory.retrieve("learning_level")
        == "beginner"
    )


def test_memory_can_update_existing_value():

    memory = Memory()

    memory.save(
        "learning_level",
        "beginner",
    )

    memory.save(
        "learning_level",
        "intermediate",
    )

    assert (
        memory.retrieve("learning_level")
        == "intermediate"
    )


def test_memory_returns_none_for_unknown_key():

    memory = Memory()

    assert (
        memory.retrieve("unknown")
        is None
    )