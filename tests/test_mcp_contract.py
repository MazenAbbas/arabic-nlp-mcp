import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from arabic_nlp_mcp.server import experimental_mcp, mcp


def test_server_exposes_typed_read_only_tools():
    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "arabic_normalize",
        "arabic_prepare_for_search",
    }
    for tool in tools:
        assert tool.output_schema
        assert tool.output_schema["additionalProperties"] is False
        assert tool.input_schema["properties"]["text"]["maxLength"] == 20_000
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False


def test_experimental_server_explicitly_adds_linguistic_tools():
    tools = asyncio.run(experimental_mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "arabic_normalize",
        "arabic_prepare_for_search",
        "arabic_detect_dialect",
        "arabic_sentiment",
        "arabic_diacritize",
    }
    for tool in tools:
        assert tool.output_schema
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False


def test_in_process_tool_call_has_structured_content():
    result = asyncio.run(mcp.call_tool("arabic_normalize", {"text": "إِنَّ"}))
    assert result.structured_content["normalized"] == "ان"


def test_search_tool_contract_and_fingerprint():
    result = asyncio.run(
        mcp.call_tool("arabic_prepare_for_search", {"text": "إِنَّ  AI", "profile": "search"})
    )
    assert result.structured_content["search_text"] == "ان ai"
    assert len(result.structured_content["fingerprint"]) == 64


def test_mcp_rejects_invalid_profile_and_oversized_text():
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("arabic_prepare_for_search", {"text": "نص", "profile": "unsafe"}))
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("arabic_prepare_for_search", {"text": "ا" * 20_001}))
