"""Tests for JobGraph and JobNode models - the DAG state machine."""

import pytest
from datetime import datetime, timezone, timedelta

from shared.job_graph.models import JobGraph, JobNode, NodeType, NodeStatus, GraphStatus


class TestJobNode:
    """Tests for individual node behavior."""

    def test_node_defaults(self):
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        assert node.status == NodeStatus.PENDING
        assert node.timeout == 120
        assert node.dependencies == []
        assert node.agent is None
        assert node.result is None
        assert node.error is None

    def test_is_terminal(self):
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        assert not node.is_terminal

        node.status = NodeStatus.COMPLETED
        assert node.is_terminal

        node.status = NodeStatus.FAILED
        assert node.is_terminal

        node.status = NodeStatus.CANCELLED
        assert node.is_terminal

        node.status = NodeStatus.RUNNING
        assert not node.is_terminal

    def test_elapsed_seconds_none_when_not_dispatched(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test")
        assert node.elapsed_seconds is None

    def test_elapsed_seconds_when_dispatched(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test")
        node.dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        elapsed = node.elapsed_seconds
        assert elapsed is not None
        assert 9.5 < elapsed < 11.0

    def test_is_timed_out(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", timeout=5)
        node.status = NodeStatus.RUNNING
        node.dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert node.is_timed_out

    def test_is_not_timed_out_when_within_limit(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", timeout=60)
        node.status = NodeStatus.RUNNING
        node.dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        assert not node.is_timed_out

    def test_is_not_timed_out_when_not_running(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", timeout=5)
        node.status = NodeStatus.COMPLETED
        node.dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert not node.is_timed_out

    def test_to_summary(self):
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="Run tests", agent="altair")
        node.status = NodeStatus.RUNNING
        summary = node.to_summary()
        assert "n1" in summary
        assert "Run tests" in summary
        assert "altair" in summary
        assert "🔄" in summary


class TestJobGraph:
    """Tests for the DAG state machine."""

    def test_graph_defaults(self):
        graph = JobGraph(goal="Test")
        assert graph.status == GraphStatus.ACTIVE
        assert graph.nodes == {}
        assert graph.thread_id is None
        assert len(graph.id) == 8  # UUID[:8]

    def test_add_node(self):
        graph = JobGraph(goal="Test", max_timeout=60)
        node = JobNode(id="n1", type=NodeType.THINK, description="think")
        graph.add_node(node)
        assert "n1" in graph.nodes
        assert graph.nodes["n1"].status == NodeStatus.READY  # No deps → immediately ready

    def test_add_node_caps_timeout(self):
        graph = JobGraph(goal="Test", max_timeout=60)
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="deploy", timeout=999)
        graph.add_node(node)
        assert graph.nodes["n1"].timeout == 60

    def test_add_node_with_dependency_stays_pending(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="first")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="second", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)
        assert graph.nodes["n1"].status == NodeStatus.READY
        assert graph.nodes["n2"].status == NodeStatus.PENDING

    def test_dependency_resolution_on_completion(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="first")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="second", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        graph.mark_completed("n1", "done")
        assert graph.nodes["n2"].status == NodeStatus.READY

    def test_dependency_resolution_on_cancel(self):
        """Cancelled nodes also satisfy dependencies."""
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="first")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="second", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        graph.cancel_node("n1")
        assert graph.nodes["n2"].status == NodeStatus.READY

    def test_multiple_dependencies_all_must_complete(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="a", agent="altair")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="b", agent="polaris")
        n3 = JobNode(id="n3", type=NodeType.RESPOND, description="c", dependencies=["n1", "n2"])
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        assert graph.nodes["n3"].status == NodeStatus.PENDING  # n2 not done

        graph.mark_running("n2")
        graph.mark_completed("n2", "done")
        assert graph.nodes["n3"].status == NodeStatus.READY

    def test_mark_running(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", agent="altair")
        graph.add_node(node)

        graph.mark_running("n1", message_id=12345)
        assert graph.nodes["n1"].status == NodeStatus.RUNNING
        assert graph.nodes["n1"].dispatched_at is not None
        assert graph.nodes["n1"].message_id == 12345

    def test_mark_completed(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        graph.add_node(node)
        graph.mark_running("n1")

        graph.mark_completed("n1", "result here")
        assert graph.nodes["n1"].status == NodeStatus.COMPLETED
        assert graph.nodes["n1"].result == "result here"
        assert graph.nodes["n1"].completed_at is not None

    def test_mark_failed(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        graph.mark_failed("n1", "connection refused")
        assert graph.nodes["n1"].status == NodeStatus.FAILED
        assert graph.nodes["n1"].error == "connection refused"

    def test_cancel_node_pending(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="first")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="second", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        assert graph.cancel_node("n2")  # PENDING → CANCELLED
        assert graph.nodes["n2"].status == NodeStatus.CANCELLED

    def test_cancel_node_ready(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        graph.add_node(node)

        assert graph.cancel_node("n1")  # READY → CANCELLED
        assert graph.nodes["n1"].status == NodeStatus.CANCELLED

    def test_cancel_node_running_fails(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        assert not graph.cancel_node("n1")  # RUNNING cannot be cancelled
        assert graph.nodes["n1"].status == NodeStatus.RUNNING

    def test_cancel_nonexistent_node(self):
        graph = JobGraph(goal="Test")
        assert not graph.cancel_node("nope")

    def test_is_goal_met_all_completed(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="a")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="b", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        assert not graph.is_goal_met()  # n2 not done

        graph.mark_running("n2")
        graph.mark_completed("n2", "sent")
        assert graph.is_goal_met()

    def test_is_goal_met_ignores_cancelled(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="a")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="b", agent="altair")
        graph.add_node(n1)
        graph.add_node(n2)

        graph.cancel_node("n2")
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        assert graph.is_goal_met()  # Only non-cancelled must be complete

    def test_is_goal_met_empty_graph(self):
        graph = JobGraph(goal="Test")
        assert not graph.is_goal_met()  # Empty graph → not met

    def test_is_goal_met_auto_sets_graph_status(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        graph.add_node(node)
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        assert graph.status == GraphStatus.COMPLETED

    def test_has_failures(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", agent="altair")
        graph.add_node(node)
        assert not graph.has_failures()

        graph.mark_running("n1")
        graph.mark_failed("n1", "error")
        assert graph.has_failures()

    def test_is_stuck(self):
        """Graph is stuck when no ready nodes, no running, and not complete."""
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="a", agent="altair")
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="b", dependencies=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        # n1 READY, n2 PENDING → not stuck
        assert not graph.is_stuck()

        # n1 RUNNING → not stuck (still waiting)
        graph.mark_running("n1")
        assert not graph.is_stuck()

        # n1 FAILED, n2 still depends on n1 → stuck!
        graph.mark_failed("n1", "error")
        assert graph.is_stuck()

    def test_is_not_stuck_when_complete(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.THINK, description="test")
        graph.add_node(node)
        graph.mark_running("n1")
        graph.mark_completed("n1", "done")
        assert not graph.is_stuck()

    def test_get_ready_nodes(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="a")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="b", agent="altair")
        n3 = JobNode(id="n3", type=NodeType.RESPOND, description="c", dependencies=["n1", "n2"])
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        ready = graph.get_ready_nodes()
        assert len(ready) == 2
        assert {n.id for n in ready} == {"n1", "n2"}

    def test_get_running_nodes(self):
        graph = JobGraph(goal="Test")
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="a", agent="altair")
        n2 = JobNode(id="n2", type=NodeType.DISPATCH, description="b", agent="polaris")
        graph.add_node(n1)
        graph.add_node(n2)

        graph.mark_running("n1")
        running = graph.get_running_nodes()
        assert len(running) == 1
        assert running[0].id == "n1"

    def test_get_timed_out_nodes(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="test", agent="altair", timeout=1)
        graph.add_node(node)
        graph.mark_running("n1")
        # Fake old dispatch time
        graph.nodes["n1"].dispatched_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        timed_out = graph.get_timed_out_nodes()
        assert len(timed_out) == 1
        assert timed_out[0].id == "n1"

    def test_remove_node(self):
        graph = JobGraph(goal="Test")
        # n1 depends on n0, so it stays PENDING (removable)
        n0 = JobNode(id="n0", type=NodeType.THINK, description="root")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="a", dependencies=["n0"])
        n2 = JobNode(id="n2", type=NodeType.RESPOND, description="b", dependencies=["n1"])
        graph.add_node(n0)
        graph.add_node(n1)
        graph.add_node(n2)

        # n1 is PENDING (blocked by n0), so remove works
        assert graph.nodes["n1"].status == NodeStatus.PENDING
        graph.remove_node("n1")
        assert "n1" not in graph.nodes
        assert graph.nodes["n2"].dependencies == []
        assert graph.nodes["n2"].status == NodeStatus.READY  # No more deps

    def test_remove_running_node_fails(self):
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.DISPATCH, description="a", agent="altair")
        graph.add_node(node)
        graph.mark_running("n1")

        graph.remove_node("n1")  # Only removes PENDING
        assert "n1" in graph.nodes  # Still there

    def test_dependency_on_nonexistent_node(self):
        """Deps on non-existent nodes are silently treated as met."""
        graph = JobGraph(goal="Test")
        node = JobNode(id="n1", type=NodeType.RESPOND, description="a", dependencies=["ghost"])
        graph.add_node(node)
        assert graph.nodes["n1"].status == NodeStatus.READY  # "ghost" skipped

    def test_get_context_for_llm(self):
        graph = JobGraph(goal="Deploy app")
        n1 = JobNode(id="n1", type=NodeType.DISPATCH, description="run build", agent="altair")
        graph.add_node(n1)
        graph.mark_running("n1")
        graph.mark_completed("n1", "build passed")

        context = graph.get_context_for_llm()
        assert "Deploy app" in context
        assert "n1" in context
        assert "completed" in context
        assert "build passed" in context

    def test_get_state_summary(self):
        graph = JobGraph(goal="Test goal")
        n1 = JobNode(id="n1", type=NodeType.THINK, description="think about it")
        graph.add_node(n1)

        summary = graph.get_state_summary()
        assert "Test goal" in summary
        assert "n1" in summary
        assert "think about it" in summary
