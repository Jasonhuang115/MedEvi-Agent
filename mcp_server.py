"""MedEvi MCP Server —— 暴露 PubMed 检索和 PICOS 提取为 MCP 工具。

启动方式:
    python mcp_server.py

Claude Desktop 配置 (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "medevi": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py"]
        }
      }
    }
"""
import json
import sys
import os

# 确保项目根在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationCapabilities
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from skills import _safe_call


server = Server("medevi-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="medevi_search",
            description="检索 PubMed 中与给定 SNP 和疾病相关的文献。返回包含标题、摘要和相关性评分的文献列表。",
            inputSchema={
                "type": "object",
                "properties": {
                    "snp": {
                        "type": "string",
                        "description": "SNP编号，如 ESR1 rs9340799",
                    },
                    "disease": {
                        "type": "string",
                        "description": "疾病名称，如 breast cancer",
                    },
                },
                "required": ["snp", "disease"],
            },
        ),
        Tool(
            name="medevi_extract",
            description="从医学文献摘要中提取 PICOS 信息和数值数据（OR/RR、95%CI、样本量等）",
            inputSchema={
                "type": "object",
                "properties": {
                    "abstract": {
                        "type": "string",
                        "description": "医学文献摘要全文",
                    }
                },
                "required": ["abstract"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "medevi_search":
        from skills.skill_search import search

        result = _safe_call(search, arguments["snp"], arguments["disease"])
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return [TextContent(type="text", text=text)]

    if name == "medevi_extract":
        from skills.skill_extract import extract_picos, extract_numerical

        picos = _safe_call(extract_picos, arguments["abstract"])
        num = _safe_call(extract_numerical, arguments["abstract"])
        result = {"picos": picos, "numerical": num}
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return [TextContent(type="text", text=text)]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationCapabilities(
                sampling=None,
                experimental=None,
                tools=None,
                prompts=None,
                resources=None,
                logging=None,
                completion=None,
            ),
            notification_options=NotificationOptions(
                tools_changed=False,
                prompts_changed=False,
                resources_changed=False,
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
