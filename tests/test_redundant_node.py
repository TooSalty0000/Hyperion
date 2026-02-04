"""Test redundant node detection and cancellation in a sequential graph.

Scenario: A 2-node sequential graph where n1's result already contains
the information n2 was supposed to produce. We detect this and cancel n2,
verifying the graph still completes correctly.
"""

import pytest

from shared.job_graph.models import (
    JobGraph, JobNode, NodeType, NodeStatus, GraphStatus,
)


def make_graph() -> JobGraph:
    """Build a 2-node sequential graph: n1 -> n2."""
    graph = JobGraph(
        goal="Fetch weather data and summarize it",
        trigger_message_id=1,
        channel_id=100,
        max_timeout=120,
    )

    n1 = JobNode(
        id="n1",
        type=NodeType.DISPATCH,
        description="Fetch today's weather data for San Francisco",
        agent="canopus",
        timeout=60,
    )
    n2 = JobNode(
        id="n2",
        type=NodeType.DISPATCH,
        description="Summarize the weather data into a short report",
        agent="altair",
        dependencies=["n1"],
        timeout=60,
    )

    graph.add_node(n1)
    graph.add_node(n2)
    return graph


def result_covers_downstream(result: str, downstream_node: JobNode) -> bool:
    """
    Heuristic check: does the result from an upstream node already cover
    what the downstream node was supposed to do?

    This is a simple keyword overlap approach. In production, Vega's LLM
    would make this judgment call via update_node / cancel_nodes tools.
    """
    description_words = set(downstream_node.description.lower().split())
    result_lower = result.lower()

    # Check if key terms from the downstream description appear in the result
    signal_words = description_words - {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "it",
        "into", "from", "with", "that", "this", "on", "at", "by",
    }
    hits = sum(1 for word in signal_words if word in result_lower)
    coverage = hits / max(len(signal_words), 1)
    return coverage >= 0.5


# -------------------------------------------------------
# Tests
# -------------------------------------------------------

class TestGraphSetup:
    """Verify the graph is wired correctly before any execution."""

    def test_initial_node_statuses(self):
        graph = make_graph()
        assert graph.nodes["n1"].status == NodeStatus.READY
        assert graph.nodes["n2"].status == NodeStatus.PENDING

    def test_n2_depends_on_n1(self):
        graph = make_graph()
        assert graph.nodes["n2"].dependencies == ["n1"]

    def test_only_n1_is_ready(self):
        graph = make_graph()
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "n1"


class TestNormalExecution:
    """Baseline: both nodes run to completion without cancellation."""

    def test_completing_n1_makes_n2_ready(self):
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "Sunny, 72F, wind 5mph")

        assert graph.nodes["n1"].status == NodeStatus.COMPLETED
        assert graph.nodes["n2"].status == NodeStatus.READY

    def test_completing_both_finishes_graph(self):
        graph = make_graph()

        graph.mark_running("n1")
        graph.mark_completed("n1", "Sunny, 72F")

        graph.mark_running("n2")
        graph.mark_completed("n2", "Weather summary: sunny and warm")

        assert graph.is_goal_met()
        assert graph.status == GraphStatus.COMPLETED


class TestRedundantNodeCancellation:
    """Core scenario: n1's result makes n2 redundant."""

    def test_rich_result_covers_downstream(self):
        """When n1 returns a full summary, n2 is detected as redundant."""
        graph = make_graph()
        graph.mark_running("n1")

        # n1 returns a result that already contains a summary + report
        rich_result = (
            "Weather report summary for San Francisco: "
            "Sunny skies, high of 72F, low of 58F, wind 5mph NW. "
            "Short report: Pleasant weather expected all day."
        )
        graph.mark_completed("n1", rich_result)

        # n2 is now READY (deps met)
        assert graph.nodes["n2"].status == NodeStatus.READY

        # Detect redundancy
        n2 = graph.nodes["n2"]
        assert result_covers_downstream(rich_result, n2)

        # Cancel n2 before it gets dispatched
        cancelled = graph.cancel_node("n2")
        assert cancelled is True
        assert graph.nodes["n2"].status == NodeStatus.CANCELLED

        # Graph should still be considered complete
        assert graph.is_goal_met()
        assert graph.status == GraphStatus.COMPLETED

    def test_sparse_result_does_not_cover_downstream(self):
        """When n1 returns minimal data, n2 is NOT redundant."""
        graph = make_graph()
        graph.mark_running("n1")

        sparse_result = "72F, sunny"
        graph.mark_completed("n1", sparse_result)

        n2 = graph.nodes["n2"]
        assert not result_covers_downstream(sparse_result, n2)

        # n2 should proceed normally
        assert graph.nodes["n2"].status == NodeStatus.READY
        assert not graph.is_goal_met()

    def test_cancel_returns_false_on_running_node(self):
        """Cannot cancel a node that is already running."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "data")
        graph.mark_running("n2")

        cancelled = graph.cancel_node("n2")
        assert cancelled is False
        assert graph.nodes["n2"].status == NodeStatus.RUNNING

    def test_graph_state_summary_after_cancel(self):
        """Verify the LLM context reflects the cancellation."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "Full summary: sunny, 72F, pleasant day")
        graph.cancel_node("n2")

        context = graph.get_context_for_llm()
        assert "completed" in context
        assert "cancelled" in context

    def test_state_summary_readable(self):
        """Verify the human-readable summary shows correct icons."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        graph.cancel_node("n2")

        summary = graph.get_state_summary()
        assert "\u2705" in summary  # completed icon
        assert "\u26d4" in summary  # cancelled icon


class TestEdgeCases:
    """Edge cases around cancellation and graph completion."""

    def test_cancel_nonexistent_node(self):
        graph = make_graph()
        assert graph.cancel_node("n99") is False

    def test_cancel_already_completed_node(self):
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        # Can't cancel a completed node
        assert graph.cancel_node("n1") is False

    def test_cancel_only_remaining_node_completes_graph(self):
        """Cancelling the last pending node after all others complete."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        graph.cancel_node("n2")

        assert graph.is_goal_met()

    def test_graph_not_stuck_after_cancel(self):
        """Cancelling n2 should not leave the graph stuck."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        graph.cancel_node("n2")

        assert not graph.is_stuck()

    def test_graph_is_stuck_when_dep_fails_without_cancel(self):
        """If n1 fails and n2 isn't cancelled, graph is stuck."""
        graph = make_graph()
        graph.mark_running("n1")
        graph.mark_failed("n1", "network error")

        # n2 stays PENDING (failed dep doesn't satisfy it)
        assert graph.nodes["n2"].status == NodeStatus.PENDING
        assert graph.is_stuck()
        assert not graph.is_goal_met()


class TestFullRedundancyWorkflow:
    """End-to-end simulation of the redundancy detection workflow."""

    def test_simulate_one_turn_completion(self):
        """
        Simulate Vega's perspective: n1 completes with rich data,
        Vega detects n2 is redundant, cancels it, and responds.
        """
        # -- Setup --
        graph = make_graph()
        assert graph.status == GraphStatus.ACTIVE

        # -- Dispatch n1 --
        ready = graph.get_ready_nodes()
        assert [n.id for n in ready] == ["n1"]
        graph.mark_running("n1")

        # -- Agent responds with rich result --
        agent_result = (
            "San Francisco Weather Report (Summary):\n"
            "Current: Sunny, 72F\n"
            "Wind: 5mph NW\n"
            "Humidity: 45%\n"
            "Short forecast report: Clear skies through evening, "
            "temperatures dropping to 58F overnight."
        )
        graph.mark_completed("n1", agent_result)

        # -- Vega evaluates: is n2 still needed? --
        n2 = graph.nodes["n2"]
        assert n2.status == NodeStatus.READY  # deps met, awaiting dispatch

        redundant = result_covers_downstream(agent_result, n2)
        assert redundant is True

        # -- Vega cancels n2 --
        graph.cancel_node("n2")
        assert n2.status == NodeStatus.CANCELLED

        # -- Graph is done --
        assert graph.is_goal_met()
        assert graph.status == GraphStatus.COMPLETED
        assert graph.nodes["n1"].result == agent_result

        # -- Verify final counts --
        completed = [n for n in graph.nodes.values() if n.status == NodeStatus.COMPLETED]
        cancelled = [n for n in graph.nodes.values() if n.status == NodeStatus.CANCELLED]
        assert len(completed) == 1
        assert len(cancelled) == 1


# -------------------------------------------------------
# Standalone runner
# -------------------------------------------------------

def _run_standalone():
    """Run as a standalone script with step-by-step output."""
    print("=" * 60)
    print("Redundant Node Detection — Standalone Simulation")
    print("=" * 60)

    # -- Build graph --
    graph = make_graph()
    print(f"\n[1] Graph created: {graph.id}")
    print(f"    Goal: {graph.goal}")
    print(f"    n1: {graph.nodes['n1'].description}")
    print(f"    n2: {graph.nodes['n2'].description} (depends on n1)")
    print(f"    n1 status: {graph.nodes['n1'].status.value}")
    print(f"    n2 status: {graph.nodes['n2'].status.value}")

    # -- Dispatch n1 --
    ready = graph.get_ready_nodes()
    print(f"\n[2] Ready nodes: {[n.id for n in ready]}")
    graph.mark_running("n1")
    print(f"    Dispatched n1 -> canopus")
    print(f"    n1 status: {graph.nodes['n1'].status.value}")
    print(f"    n2 status: {graph.nodes['n2'].status.value}")

    # -- Simulate agent response with rich result --
    agent_result = (
        "San Francisco Weather Report (Summary):\n"
        "Current: Sunny, 72F\n"
        "Wind: 5mph NW\n"
        "Humidity: 45%\n"
        "Short forecast report: Clear skies through evening, "
        "temperatures dropping to 58F overnight."
    )
    graph.mark_completed("n1", agent_result)
    print(f"\n[3] Agent (canopus) responded:")
    print(f"    Result: {agent_result[:80]}...")
    print(f"    n1 status: {graph.nodes['n1'].status.value}")
    print(f"    n2 status: {graph.nodes['n2'].status.value}")

    # -- Redundancy check --
    n2 = graph.nodes["n2"]
    redundant = result_covers_downstream(agent_result, n2)
    print(f"\n[4] Redundancy check for n2:")
    print(f"    n2 description: \"{n2.description}\"")
    print(f"    Result covers n2? {redundant}")

    if redundant:
        graph.cancel_node("n2")
        print(f"    -> Cancelled n2 (result already contains summary/report)")
        print(f"    n2 status: {graph.nodes['n2'].status.value}")
    else:
        print(f"    -> n2 still needed, would dispatch to altair")

    # -- Final state --
    print(f"\n[5] Final graph state:")
    print(f"    Graph status: {graph.status.value}")
    print(f"    Goal met: {graph.is_goal_met()}")
    print(f"    Stuck: {graph.is_stuck()}")
    for node in graph.nodes.values():
        icon = {"completed": "OK", "cancelled": "SKIP", "running": "...", "pending": "--", "ready": ">>", "failed": "XX"}
        print(f"    [{icon.get(node.status.value, '??')}] {node.id}: {node.status.value}")
        if node.result:
            print(f"         result: {node.result[:60]}...")

    print(f"\n{'=' * 60}")

    # -- Now show what happens WITHOUT redundancy detection --
    print("\n\nComparison: sparse result (no redundancy)")
    print("-" * 40)
    graph2 = make_graph()
    graph2.mark_running("n1")
    sparse = "72F, sunny"
    graph2.mark_completed("n1", sparse)
    redundant2 = result_covers_downstream(sparse, graph2.nodes["n2"])
    print(f"    n1 result: \"{sparse}\"")
    print(f"    Covers n2? {redundant2}")
    print(f"    n2 status: {graph2.nodes['n2'].status.value} (would be dispatched next)")
    print(f"    Goal met: {graph2.is_goal_met()}")

    print(f"\nDone.")


if __name__ == "__main__":
    _run_standalone()
