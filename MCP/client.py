import asyncio
import json
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

DEFAULT_NOTEBOOK_PATH = "notebooks/test.ipynb"  # Change as needed

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages = []  # conversational history with tool calls

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools

        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def run_prompt(self, prompt: str):
        # Initialize message history
        self.messages = [{"role": "user", "content": prompt}]

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in self.tools
        ]

        while True:
            print("\n🤖 Calling OpenAI to process prompt...")
            response = self.openai.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=self.messages,
                tools=tool_defs,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls]
                })

                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    # Inject default file_path if tool expects it and it's missing
                    if "path" in tool_call.function.arguments and "path" not in args:
                        args["path"] = DEFAULT_NOTEBOOK_PATH

                    print(f"🔧 Tool call: {name}({args})")
                    result = await self.session.call_tool(name, args)

                    print(f"📦 Tool result: {result.content}")

                    # Send result back to LLM
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result.content
                    })
            else:
                print("\n✅ Final LLM response:")
                print(message.content)
                break

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])

        while True:
            query = input("\n💬 Prompt: ").strip()
            if query.lower() in {"quit", "exit"}:
                break
            await client.run_prompt(query)

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
