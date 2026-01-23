"""Tests for ToolParameter and ToolDefinition JSON schema generation."""

import pytest

from shared.llm.types import ToolParameter, ToolDefinition


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_basic_string_param(self):
        p = ToolParameter(name="msg", type="string", description="A message")
        assert p.name == "msg"
        assert p.type == "string"
        assert p.required is True
        assert p.items is None
        assert p.properties is None

    def test_optional_param(self):
        p = ToolParameter(name="opt", type="string", description="Optional", required=False)
        assert p.required is False

    def test_enum_param(self):
        p = ToolParameter(name="status", type="string", description="Status", enum=["a", "b", "c"])
        assert p.enum == ["a", "b", "c"]

    def test_array_with_items(self):
        items = {"type": "string", "description": "An item"}
        p = ToolParameter(name="tags", type="array", description="Tags", items=items)
        assert p.items == items

    def test_object_with_properties(self):
        props = {"name": {"type": "string"}, "age": {"type": "integer"}}
        p = ToolParameter(name="user", type="object", description="User", properties=props)
        assert p.properties == props


class TestToolDefinitionSchema:
    """Tests for ToolDefinition.to_json_schema()."""

    def test_simple_params(self):
        defn = ToolDefinition(
            name="greet",
            description="Say hello",
            parameters=[
                ToolParameter(name="name", type="string", description="Who to greet"),
            ]
        )
        schema = defn.to_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]

    def test_optional_not_in_required(self):
        defn = ToolDefinition(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="required_param", type="string", description="required"),
                ToolParameter(name="optional_param", type="string", description="optional", required=False),
            ]
        )
        schema = defn.to_json_schema()
        assert "required_param" in schema["required"]
        assert "optional_param" not in schema["required"]

    def test_enum_in_schema(self):
        defn = ToolDefinition(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="color", type="string", description="Color", enum=["red", "blue"]),
            ]
        )
        schema = defn.to_json_schema()
        assert schema["properties"]["color"]["enum"] == ["red", "blue"]

    def test_default_in_schema(self):
        defn = ToolDefinition(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="count", type="integer", description="Count", default=10),
            ]
        )
        schema = defn.to_json_schema()
        assert schema["properties"]["count"]["default"] == 10

    def test_array_with_items_in_schema(self):
        items_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["id"],
        }
        defn = ToolDefinition(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="items", type="array", description="List of items", items=items_schema),
            ]
        )
        schema = defn.to_json_schema()
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"] == items_schema
        assert schema["properties"]["items"]["items"]["properties"]["id"]["type"] == "string"

    def test_object_with_properties_in_schema(self):
        props = {"name": {"type": "string"}, "age": {"type": "integer"}}
        defn = ToolDefinition(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="user", type="object", description="A user", properties=props),
            ]
        )
        schema = defn.to_json_schema()
        assert schema["properties"]["user"]["type"] == "object"
        assert schema["properties"]["user"]["properties"] == props

    def test_no_params(self):
        defn = ToolDefinition(name="noop", description="Does nothing")
        schema = defn.to_json_schema()
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_graph_tools_node_schema(self):
        """Verify the actual NODE_ITEM_SCHEMA used in graph tools."""
        from vega.tools.graph import NODE_ITEM_SCHEMA, CreatePlanTool

        tool = CreatePlanTool()
        schema = tool.to_definition().to_json_schema()

        # nodes param should have items
        nodes_schema = schema["properties"]["nodes"]
        assert nodes_schema["type"] == "array"
        assert "items" in nodes_schema

        # items should describe the node object
        item = nodes_schema["items"]
        assert item["type"] == "object"
        assert "id" in item["properties"]
        assert "type" in item["properties"]
        assert "description" in item["properties"]
        assert "agent" in item["properties"]
        assert "dependencies" in item["properties"]
        assert "timeout" in item["properties"]

        # type should have enum
        assert item["properties"]["type"]["enum"] == ["think", "dispatch", "respond"]
        assert item["properties"]["agent"]["enum"] == ["altair", "polaris", "canopus"]

        # required fields
        assert "id" in item["required"]
        assert "type" in item["required"]
        assert "description" in item["required"]
