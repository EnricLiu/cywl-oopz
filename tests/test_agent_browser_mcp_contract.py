from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FIXTURE = Path(__file__).parent / "fixtures" / "agent_browser_mcp_v0_33_contract.json"


def load_contract() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def test_agent_browser_contract_fixture_has_an_explicit_project_allow_list() -> None:
    contract = load_contract()
    project_tools = set(contract["project_tools"])
    never_expose = set(contract["never_expose"])

    assert project_tools
    assert "agent_browser_eval" in never_expose
    assert project_tools.isdisjoint(never_expose)
    assert all(name.startswith("agent_browser_") for name in project_tools | never_expose)


@pytest.mark.asyncio
async def test_agent_browser_mcp_contract_and_lifecycle_when_explicitly_enabled() -> None:
    if os.getenv("CYWL_RUN_AGENT_BROWSER_MCP_TESTS") != "1":
        pytest.skip("set CYWL_RUN_AGENT_BROWSER_MCP_TESTS=1 to run the agent-browser MCP contract")
    executable = shutil.which("agent-browser")
    if executable is None:
        pytest.fail("agent-browser is not available on PATH")

    contract = load_contract()
    server_contract = contract["server"]
    project_tools = contract["project_tools"]
    unique = uuid4().hex[:8]
    session_name = f"cw0-{unique}"
    namespace = f"cywl-w0-{unique}"
    environment = dict(os.environ)
    environment.update(
        {
            "AGENT_BROWSER_SESSION": session_name,
            "AGENT_BROWSER_NAMESPACE": namespace,
            "AGENT_BROWSER_CONTENT_BOUNDARIES": "true",
            "AGENT_BROWSER_MAX_OUTPUT": "12000",
            "AGENT_BROWSER_IDLE_TIMEOUT_MS": "2000",
        }
    )
    parameters = StdioServerParameters(
        command=executable,
        args=["mcp", "--tools", server_contract["profile"]],
        env=environment,
        cwd=str(Path.cwd()),
    )
    common = {
        "session": session_name,
        "namespace": namespace,
        "timeoutMs": 20_000,
    }
    session_open = False

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=30),
        ) as client:
            initialized = await client.initialize()
            assert initialized.protocolVersion == server_contract["protocol"]
            assert initialized.serverInfo.name == server_contract["name"]
            assert initialized.serverInfo.version == server_contract["version"]

            tools = []
            cursor = None
            while True:
                page = await client.list_tools(cursor)
                tools.extend(page.tools)
                cursor = page.nextCursor
                if cursor is None:
                    break
            discovered = {tool.name: tool for tool in tools}

            for name, expected in project_tools.items():
                assert name in discovered
                schema = discovered[name].inputSchema
                assert schema["additionalProperties"] is False
                assert schema.get("required", []) == expected["required"]
                for property_name, property_type in expected["properties"].items():
                    assert schema["properties"][property_name]["type"] == property_type

            assert "agent_browser_eval" in discovered
            assert "agent_browser_eval" not in project_tools

            try:
                opened = await client.call_tool(
                    "agent_browser_open",
                    {"url": "https://example.com", **common},
                )
                assert not opened.isError
                assert opened.structuredContent["response"]["success"] is True
                session_open = True

                snapshot = await client.call_tool(
                    "agent_browser_snapshot",
                    {
                        "interactive": False,
                        "compact": True,
                        **common,
                    },
                )
                assert not snapshot.isError
                snapshot_data = snapshot.structuredContent["response"]["data"]
                assert "Example Domain" in snapshot_data["snapshot"]

                document = await client.call_tool("agent_browser_read", common)
                assert not document.isError
                document_data = document.structuredContent["response"]["data"]
                assert document_data["finalUrl"] == "https://example.com/"
                assert "Example Domain" in document_data["content"]
            finally:
                if session_open:
                    closed = await client.call_tool("agent_browser_close", common)
                    assert not closed.isError
                    assert closed.structuredContent["response"]["data"]["closed"] is True

    doctor_output = b""
    for _ in range(50):
        doctor = await asyncio.create_subprocess_exec(
            executable,
            "doctor",
            "--offline",
            "--quick",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        doctor_output, _ = await asyncio.wait_for(doctor.communicate(), timeout=10)
        assert doctor.returncode == 0
        if b"No active daemons" in doctor_output:
            break
        await asyncio.sleep(0.1)
    assert b"No active daemons" in doctor_output
