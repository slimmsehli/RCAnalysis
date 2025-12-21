import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI (Make sure your API key is in your environment variables)
client = OpenAI()

import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = OpenAI() #(api_key="your-api-key")

async def run_intelligent_agent():
    server_params = StdioServerParameters(command="python", args=["server_test.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            system_msg = """You are a Verification Engineer. When you search logs:
                        1. Look for patterns like 'filename.sv(line)' or 'at path/to/file.sv line X and ask for full file path'.
                        2. If you find an error, you MUST identify the file and line number.
                        3. If the file path is not on the error line, look at the 5 lines above it."""
            system_msg2 = "You are an RTL Debugger. Use the search_log_keyword tool to find errors and investigate their context."
            # The conversation history starts here
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Start by searching for the first UVM_ERROR in 'sim.log' and tell me what happened."}
            ]


            # 1. Ask the LLM what it wants to do
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search_log_keyword",
                            "description": "Search the simulation log for errors, warnings, or UVM messages.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "log_path": {"type": "string"},
                                    "keyword": {"type": "string", "description": "e.g., 'UVM_ERROR' or 'FATAL'"},
                                    "context_lines": {"type": "integer", "default": 10}
                                },
                                "required": ["log_path", "keyword"]
                            }
                        }
                    }
                ]
            )

            # 2. Check if the LLM wants to call a tool
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call:
                import json
                args = json.loads(tool_call.function.arguments)
                print(f"[LLM Action] Searching for: {args['keyword']}")

                # 3. Execute the Python Tool
                result = await session.call_tool("search_log_keyword", {
                    "log_path": "sim.log",
                    "keyword": args["keyword"],
                    "context_lines": args.get("context_lines", 10)
                })

                # 4. Feed the result back to the LLM
                messages.append(response.choices[0].message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content[0].text
                })

                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                print("\n[LLM Final Analysis]:\n", final_response.choices[0].message.content)

if __name__ == "__main__":
    print(" [INFO] Running the engine ...")
    asyncio.run(run_intelligent_agent())