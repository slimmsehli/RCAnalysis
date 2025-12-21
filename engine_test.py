import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

async def run_agent_loop():
    server_params = StdioServerParameters(command="python", args=["server_test.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Setup the Agent's "Memory" and specialized instructions
            # Wrap the dictionary in a list [ ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Silicon Debug Engineer. Your goal is to find the root cause of failures using logs, source, and VCD.\n\n"
                        "DEBUGGING PROTOCOL:\n"
                        "1. **Log Analysis**: Search the log for 'UVM_ERROR' or 'FATAL'. Identify the file, line number, and simulation TIMESTAMP.\n"
                        "2. **Source Inspection**: Read the source code around that line. Identify the signals involved in the failing logic.\n"
                        "3. **Waveform Correlation**: \n"
                        "   - Use 'vcd_get_timescale' to understand the simulation units.\n"
                        "   - Map source signals to VCD hierarchy. If a signal 'out' is in 'top.dut', query 'top.dut.out'.\n"
                        "   - Use 'vcd_get_signal_value_at_timestamp' to check the actual hardware state at the time of failure.\n"
                        "4. **Root Cause**: Compare the 'Expected' value (from source logic) vs 'Actual' value (from VCD). Explain the discrepancy."
                    )
                },
                {
                    "role": "user",
                    "content": "Investigate the failure in 'sim.log' and explain the code error."
                }
            ]

            # 2. Define the tools for OpenAI
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_log_keyword",
                        "description": "Search log for keywords and get surrounding context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "log_path": {"type": "string"},
                                "keyword": {"type": "string"},
                                "context_lines": {"type": "integer"}
                            },
                            "required": ["log_path", "keyword"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_source_snippet",
                        "description": "Read a specific portion of a source code file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "line_number": {"type": "integer"},
                                "context": {"type": "integer"}
                            },
                            "required": ["file_path", "line_number"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_log_keyword",
                        "description": "Search the simulation log for errors, warnings, or specific keywords.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "log_path": {"type": "string"},
                                "keyword": {"type": "string"},
                                "context_lines": {"type": "integer", "default": 10}
                            },
                            "required": ["log_path", "keyword"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_source_snippet",
                        "description": "Extract a portion of the Verilog/SystemVerilog source code.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "line_number": {"type": "integer"},
                                "context": {"type": "integer", "default": 5}
                            },
                            "required": ["file_path", "line_number"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_get_simulation_time",
                        "description": "Returns the total simulation time (duration) of the VCD file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to the VCD file."}
                            },
                            "required": ["path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_get_timescale",
                        "description": "Returns the magnitude and unit of the simulation timescale (e.g., 1 ns).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_get_signal_value_at_timestamp",
                        "description": "Get the value of a specific signal at a precise timestamp.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string",
                                                "description": "Full hierarchical name, e.g., 'top.dut.clk'"},
                                "timestamp": {"type": "number", "description": "The simulation time to query."}
                            },
                            "required": ["path", "signal_name", "timestamp"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_get_signal_values_in_timeframe",
                        "description": "Returns all signal changes and values within a specified time window.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string"},
                                "start": {"type": "number"},
                                "end": {"type": "number"}
                            },
                            "required": ["path", "signal_name", "start", "end"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_count_signal_all_transitions",
                        "description": "Counts the number of signal edges (posedge/negedge) in a time window.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string"},
                                "edge": {"type": "string", "enum": ["posedge", "negedge", "both"]},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "bit_index": {"type": "integer",
                                              "description": "Optional bit index if the signal is a bus."}
                            },
                            "required": ["path", "signal_name", "edge", "start", "end"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_next_change_after",
                        "description": "Find the very next time a signal changes after a given timestamp.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string"},
                                "timestamp": {"type": "number"}
                            },
                            "required": ["path", "signal_name", "timestamp"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_prev_change_before",
                        "description": "Find the most recent time a signal changed before a given timestamp.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string"},
                                "timestamp": {"type": "number"}
                            },
                            "required": ["path", "signal_name", "timestamp"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "vcd_search_value",
                        "description": "Searches for occurrences of a specific value on a signal within a time range.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "signal_name": {"type": "string"},
                                "value": {"type": "string",
                                          "description": "The value to search for (e.g. '1', '0', '4'hAF')."},
                                "start": {"type": "number"},
                                "end": {"type": "number"}
                            },
                            "required": ["path", "signal_name", "value"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_vcd_signals",
                        "description": "List all signal names in the VCD. Use a pattern to filter (e.g. 'dut').",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "pattern": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ]

            # 3. The ReAct Loop
            max_iterations = 15
            for i in range(max_iterations):
                print(f"\n[Iteration {i + 1}] Thinking...")

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                messages.append(response_message)

                # Check if the LLM wants to finish or call a tool
                if not response_message.tool_calls:
                    print("\n--- Final Diagnosis ---")
                    print(response_message.content)
                    break

                # Handle tool calls
                for tool_call in response_message.tool_calls:
                    fname = tool_call.function.name
                    fargs = json.loads(tool_call.function.arguments)

                    # LOGGING FOR YOU TO TRACK PROGRESS
                    print(f"[Iteration {i + 1}] LLM calls {fname}, fargs {fargs}")

                    # Execute the MCP tool (which calls your vcdvcd wrapper)
                    result = await session.call_tool(fname, fargs)
                    #print(f"[Iteration {i + 1}] LLM results from function {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fname,
                        "content": result.content[0].text
                    })

if __name__ == "__main__":
    asyncio.run(run_agent_loop())