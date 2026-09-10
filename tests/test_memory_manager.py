# tests/test_memory_manager.py

from agents.memory import Memory
from agents.memory_manager import MemoryManager


def test_memory_manager_starts_with_memory():

    manager = MemoryManager()

    assert manager.memory is not None


def test_memory_manager_can_save_memory():

    manager = MemoryManager()

    manager.save(
        "user_goal",
        "AI Engineer",
    )

    assert (
        manager.retrieve("user_goal")
        == "AI Engineer"
    )


def test_memory_manager_can_retrieve_unknown_memory():

    manager = MemoryManager()

    assert (
        manager.retrieve("unknown")
        is None
    )


def test_memory_manager_can_use_existing_memory():

    memory = Memory()

    memory.save(
        "user_goal",
        "AI Engineer",
    )

    manager = MemoryManager(
        memory=memory
    )

    assert (
        manager.retrieve("user_goal")
        == "AI Engineer"
    )


def test_memory_manager_can_update_memory():

    manager = MemoryManager()

    manager.save(
        "learning_level",
        "beginner",
    )

    manager.save(
        "learning_level",
        "intermediate",
    )

    assert (
        manager.retrieve("learning_level")
        == "intermediate"
    )


def test_memory_manager_builds_context():

    manager = MemoryManager()

    manager.save(
        "user_goal",
        "AI Engineer",
    )

    context = manager.build_context(
        "user_goal"
    )

    assert context == [
        {
            "role": "system",
            "content": (
                "Relevant memory:\n"
                "AI Engineer"
            ),
        }
    ]


def test_memory_manager_returns_empty_context_for_unknown_memory():

    manager = MemoryManager()

    context = manager.build_context(
        "unknown"
    )

    assert context == []