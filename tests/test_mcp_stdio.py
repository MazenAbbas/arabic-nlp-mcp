import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_real_stdio_handshake_listing_and_structured_call():
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "arabic_nlp_mcp.server"],
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream, read_timeout_seconds=10) as session,
        ):
            await session.initialize()
            listing = await session.list_tools()
            assert {tool.name for tool in listing.tools} == {
                "arabic_normalize",
                "arabic_prepare_for_search",
            }
            result = await session.call_tool("arabic_prepare_for_search", {"text": "إِنَّ AI رقم ١٢"})
            assert result.is_error is False
            assert result.structured_content["search_text"] == "ان ai رقم 12"
            rejected = await session.call_tool(
                "arabic_prepare_for_search", {"text": "ا" * 20_001}
            )
            assert rejected.is_error is True
            recovered = await session.call_tool("arabic_normalize", {"text": "ســلام"})
            assert recovered.is_error is False
            assert recovered.structured_content["normalized"] == "سلام"

    asyncio.run(exercise())
