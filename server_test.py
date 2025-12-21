import bisect

from mcp.server.fastmcp import FastMCP
# function to get information from vcd file
import vcd_getinfo as getinfo
from typing import Any, Dict, List, Optional, Tuple, Union
from vcdvcd import VCDVCD
from decimal import Decimal
import re
import os

mcp = FastMCP("RTL_Toolbox")

# This is an example to load a log file,
# extract the output vcd file name and then get the timescale form the vcd file


##########################################
###### parse a log file and extract errors
##########################################
@mcp.tool()
def parse_log_for_errors(log_path: str) -> str:
    """Finds the VCD file and any Verilog errors in the log."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        vcd_file = "unknown"
        found_errors = []

        for line in lines:
            # 1. Look for the VCD filename anywhere in the log
            vcd_match = re.search(r"dumped in ([\w\.]+) file", line)
            if vcd_match:
                vcd_file = vcd_match.group(1)

            # 2. Look for the Error pattern
            # Matches: ERROR: path/to/file.sv at line 123
            error_match = re.search(r"ERROR:\s+([\d\w\.\/]+)\s+at\s+line\s+(\d+)", line)
            if error_match:
                found_errors.append({
                    "file": error_match.group(1),
                    "line": error_match.group(2),
                    "vcd": vcd_file
                })

        if not found_errors:
            return "No errors found."

        # Format output for the Agent
        output = []
        for err in found_errors:
            output.append(f"Found Error in {err['file']} at line {err['line']}. Relevant VCD: {err['vcd']}")

        return "\n".join(output)

    except Exception as e:
        return f"File error: {str(e)}"

    # Always return a string for the content[0].text field
    return str(errors) if errors else "[]"

##########################################
###### extract a code snipet from a source code file
##########################################
@mcp.tool()
def get_source_snippet(file_path: str, line_number: int, context: int = 5):
    """Extracts a specific line from source code with surrounding context."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            start = max(0, line_number - context - 1)
            end = min(len(lines), line_number + context)
            return "".join(lines[start:end])
    except FileNotFoundError:
        return f"File {file_path} not found."

##########################################
###### find the Nth erros message in the log file
##########################################
@mcp.tool()
def find_first_uvm_error(log_path: str) -> str:
    """Finds the line number of the first UVM_ERROR or UVM_FATAL."""
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if "UVM_ERROR" in line or "UVM_FATAL" in line:
                return str(i + 1) # Return as string for MCP
    return "0"

##########################################
###### extract the previous 20 lines
##########################################
# @TODO to change this to extrat N lines before the error line
@mcp.tool()
def get_error_context(log_path: str, error_line: int, window: int = 20) -> str:
    """Extracts the 'N' lines preceding an error for LLM analysis."""
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()

        # Adjust for 0-based indexing
        idx = error_line - 1
        start = max(0, idx - window)
        # We include the error line itself (+1)
        snippet = all_lines[start: idx + 1]

        return "".join(snippet)
    except Exception as e:
        return f"Error extracting context: {e}"

##########################################
###### this function is to search for any keyword in a file and extract some lines around the erro
##########################################
@mcp.tool()
def search_log_keyword(log_path: str, keyword: str, context_lines: int = 10) -> str:
    """
    Searches for a keyword in the log and returns the matching line
    plus surrounding context. Useful for finding UVM_ERRORs and file paths.
    """
    try:
        if not os.path.exists(log_path):
            return f"Error: Log file '{log_path}' not found."

        with open(log_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "".join(lines[start:end])

                # We return the snippet and a hint for the LLM
                return (f"--- Results for '{keyword}' near line {i + 1} ---\n"
                        f"{snippet}\n"
                        "--- End of Snippet ---")

        return f"Keyword '{keyword}' not found in the log."
    except Exception as e:
        return f"Error: {str(e)}"


##########################################
###### extract code snipet from a source code file
##########################################
@mcp.tool()
def get_source_snippet(file_path: str, line_number: int, context: int = 5) -> str:
    """
    Opens a source code file and extracts the code around a specific line.
    """
    try:
        # Standardize path (handles ./simulation/top.sv)
        normalized_path = os.path.normpath(file_path)

        if not os.path.exists(normalized_path):
            return f"Error: Source file '{normalized_path}' not found."

        with open(normalized_path, "r") as f:
            lines = f.readlines()

        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        snippet = "".join(lines[start:end])

        return f"--- Source: {normalized_path} (Lines {start + 1} to {end}) ---\n{snippet}"
    except Exception as e:
        return f"Error reading source file: {str(e)}"

############################################## VCD parsing

######################################################################
###### time parsing helpers ----
######################################################################

_UNIT_TO_SEC = {
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15,
}

def _to_seconds(t: Union[str, float, int]) -> float:
    if isinstance(t, (int, float)): return float(t)
    s = str(t).strip()
    m = re.match(r"^\s*([0-9]+(\.[0-9]+)?)\s*([a-zA-Z]+)\s*$", s)
    if m:
        val = float(m.group(1))
        unit = m.group(3).lower()
        if unit not in _UNIT_TO_SEC:
            raise ValueError(f"Unknown time unit '{unit}' in '{t}'")
        return val * _UNIT_TO_SEC[unit]
    return float(s)  # assume seconds

def _in_window(tv: List[Tuple[int, str]], start_s: Optional[float], end_s: Optional[float]) -> List[Tuple[float, str]]:
    """Filter tv changes between start and end (given in seconds). `vcdvcd` stores times in raw units;
       we don't know the timescale multiplier here, so treat times as seconds already.
       If your simulator uses integer timesteps as raw units (e.g. ns), pass values consistently.
    """
    # vcdvcd gives times as int (raw) — we will treat them as seconds unless you rescale externally.
    s = -float("inf") if start_s is None else start_s
    e = float("inf") if end_s is None else end_s
    return [(float(t), v) for (t, v) in tv if s <= float(t) <= e]

def _bit(value: str, size_hint: Optional[int], bit_index: Optional[int]) -> Optional[int]:
    """Return scalar bit 0/1 from a scalar ('0','1','x','z') or binary string for vectors."""
    v = value.lower()
    if v in ("0", "1"): return int(v)
    if v in ("x", "z"): return None
    # vector like '1010'
    if bit_index is None: return None
    if not all(ch in "01xz" for ch in v): return None
    # LSB is rightmost
    if bit_index < 0 or bit_index >= len(v): return None
    ch = v[-1 - bit_index]
    if ch in ("0", "1"): return int(ch)
    return None

######################################################################
###### get simulation time
######################################################################
def vcd_get_simulation_time(path: str, store_scopes: bool = False) -> float:
    """
    Return total simulation time in seconds (float).
    Accepts a VCD file path.
    """
    if path.exists():
        vcd = VCDVCD(path, store_tvs=True, store_scopes=store_scopes)
        magnitude, unit = getinfo.get_timescale(vcd)
    else:
        return 0
    # Parse or reuse object // old function used to read the vcd file
    # vcd = VCDVCD(vcd_input, store_tvs=True) if isinstance(vcd_input, str) else vcd_input

    # Timescale -> seconds per unit as Decimal
    ts = vcd.timescale
    magnitude = Decimal(ts["magnitude"])
    unit = ts["unit"].lower()
    unit_to_sec = {
        "s":  Decimal("1"),
        "ms": Decimal("1e-3"),
        "us": Decimal("1e-6"),
        "ns": Decimal("1e-9"),
        "ps": Decimal("1e-12"),
        "fs": Decimal("1e-15"),
    }
    if unit not in unit_to_sec:
        raise ValueError(f"Unsupported VCD timescale unit: {unit}")
    seconds_per_unit = magnitude * unit_to_sec[unit]

    # Find the largest timestamp across all signals
    # vcd.signals is a list of full signal names; vcd[ref].tv -> [(time_raw, value), ...]
    max_time_raw = Decimal(0)
    for ref in getattr(vcd, "signals", []):
        sig = vcd[ref]
        tv = getattr(sig, "tv", [])
        if tv:
            # last change time (raw units, typically int)
            last_t = Decimal(tv[-1][0])
            if last_t > max_time_raw:
                max_time_raw = last_t

    # Convert to seconds & return as float
    sim_time_seconds = max_time_raw * seconds_per_unit
    return float(sim_time_seconds)

##########################################
###### Get the timescale from a vcd file
##########################################
from pathlib import Path
@mcp.tool()
def vcd_get_timescale(path: str, store_scopes: bool = False) -> str:
    """ Get the magnitude and the unit of the timescale of a vcd file and output as two string format """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=store_scopes)
        magnitude, unit = getinfo.get_timescale(vcdobj)
        return f"{magnitude} {unit}"
    else:
        return "File does not exist"


######################################################################
###### get signal value at a specific timestamp
######################################################################

def vcd_get_signal_value_at_timestamp(path: str, signal_name: str, timestamp: Union[str, float, int], method: str = "previous") -> Any:
    """
    Return the value of a signal at a specific timestamp.
    the input is the signal name and the timestamp
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return 0, 0
    tv = vcdobj[signal_name].tv  # list of (time, value) tuples. [2](https://github.com/cirosantilli/vcdvcd)
    if not tv: return None
    t_sec = _to_seconds(timestamp)
    times = [float(t) for (t, _) in tv]
    idx = bisect.bisect_right(times, t_sec) - 1
    if method == "exact":
        i = bisect.bisect_left(times, t_sec)
        if 0 <= i < len(times) and times[i] == t_sec:
            return tv[i][1]
        return None
    if idx < 0:
        return None
    return tv[idx][1]

######################################################################
###### get signal value at a specific time frame
######################################################################

def vcd_get_signal_values_in_timeframe(path: str, signal_name: str, start: Optional[Union[str, float, int]], end: Optional[Union[str, float, int]], include_start_prev: bool = True) -> List[Tuple[float, Any]]:
    """
    Return the values of a signal at a specific time window.
    the input is the signal name and the time window high and low limit
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return [(0, 0)]
    tv = vcdobj[signal_name].tv
    s = None if start is None else _to_seconds(start)
    e = None if end is None else _to_seconds(end)
    window = _in_window(tv, s, e)
    out = []
    if include_start_prev and start is not None:
        prev = vcd_get_signal_value_at_timestamp(vcdobj, signal_name, s, method="previous")
        if prev is not None:
            out.append((s, prev))
    out.extend(window)
    return out

######################################################################
###### Count a signal transitions in a time frame
######################################################################
# count_signal_all_transitions (vcd: VCDVCD, signal_name: str, edge: str, start:etr, end:str, bit_index:int)
def vcd_count_signal_all_transitions(path: str, signal_name: str, edge: str, start: Optional[Union[str, float, int]], end: Optional[Union[str, float, int]], bit_index: Optional[int] = None) -> int:
    """
    Return the count of the number of a signal edges in a time window
    the input is the signal name, the edge, the start and finsh time limit of the time window and a bit index if it is a bus
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return 0
    tv = vcdobj[signal_name].tv
    s = None if start is None else _to_seconds(start)
    e = None if end is None else _to_seconds(end)
    window = _in_window(tv, s, e)
    if not window: return 0
    last = _bit(window[0][1], None, bit_index)
    cnt = 0
    for _, v in window[1:]:
        cur = _bit(v, None, bit_index)
        if last is None or cur is None:
            last = cur
            continue
        if edge == "rising" and last == 0 and cur == 1:
            cnt += 1
        elif edge == "falling" and last == 1 and cur == 0:
            cnt += 1
        last = cur
    return cnt

######################################################################
###### Get the first edge after a timestamp
######################################################################
### @TODO need add the first value and the value after the transition in the output results
def vcd_next_change_after(path: str, signal_name: str, timestamp: Union[str, float, int]) -> Optional[Tuple[float, Any]]:
    """
    Return the first change of a signal after a timestamp.
    the input is the signal name and the timestamp
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return 0, 0
    tv = vcdobj[signal_name].tv
    if not tv: return None
    t_sec = _to_seconds(timestamp)
    times = [float(t) for (t, _) in tv]
    idx = bisect.bisect_right(times, t_sec)
    if idx < len(tv):
        t, v = tv[idx]
        return (float(t), v)
    return None

######################################################################
###### Get the first edge after a timestamp
######################################################################
### @TODO need add the first value and the value after the transition in the output results
def vcd_prev_change_before(path: str, signal_name: str, timestamp: Union[str, float, int]) -> Optional[Tuple[float, Any]]:
    """
    Return the first change of a signal before a timestamp.
    the input is the signal name and the timestamp
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return 0, 0
    tv = vcdobj[signal_name].tv
    if not tv: return None
    t_sec = _to_seconds(timestamp)
    times = [float(t) for (t, _) in tv]
    idx = bisect.bisect_left(times, t_sec) - 1
    if idx >= 0:
        t, v = tv[idx]
        return (float(t), v)
    return None

######################################################################
###### Search for a vlaue of a singal
######################################################################
### @TODO  need to add the entire timeframe where the signal has the extracted value not only the transition
def vcd_search_value(path: str, signal_name: str, value: Any, start: Optional[Union[str, float, int]] = None, end: Optional[Union[str, float, int]] = None) -> List[float]:
    """
    Return if a signal has encountered a specific value during the simulation
    the input is the signal name and the value to be searched
    """
    if path.exists():
        vcdobj = VCDVCD(path, store_tvs=True, store_scopes=False)
    else:
        return [0]
    changes = vcd_get_signal_values_in_timeframe(vcdobj, signal_name, start, end, include_start_prev=False)
    target = value.lower() if isinstance(value, str) else value
    return [t for (t, v) in changes if (v.lower() if isinstance(v, str) else v) == target]

######################################################################
###### show the hierarchy of the design
######################################################################
@mcp.tool()
def list_vcd_signals(path: str, pattern: str = "", store_scopes: bool = False) -> str:
    """Returns a list of all signals in the VCD matching a pattern (e.g., 'dut')."""
    try:
        if path.exists():
            vcdobj = VCDVCD(path, store_tvs=True, store_scopes=store_scopes)
        else :
            return "error : vcd File does not exist"
        # Assuming vcdvcd, we can list the keys (signals)
        signals = [s for s in vcdobj.signals if pattern in s]
        return "\n".join(signals[:20]) # Limit to 20 so we don't blow the token limit
    except Exception as e:
        return f"Error: {e}"

######################################################################
######  search for multiple signals in a timeframe and return their values
######################################################################
# take a list of signals, a window timeframe, and give back a table of the signals and their transitions int eh given window
# @TODO not implemented yet


######################################################################
######  search for a state of a signal in a timeframe
######################################################################
# need to implement the search for a state and values for a signal in a timeframe and get back the time
# Logic-Based Search
# @TODO not implemented yet


######################################################################
######  stable interval search for a signal in a timeframe
######################################################################
# to search for how long a signal is stable without gliches for setup and hold time
# Protocol-Specific Analysis
# @TODO not implemented yet

######################################################################
######  map logic values to states like IDLE, ready and others with connection to a source code or a package
######################################################################
# Bus & Enum Interpretation
# @TODO not implemented yet


######################################################################
######  give the related signals in a module based on the input signal
######################################################################
# Hierarchy Navigation
# @TODO not implemented yet


######################################################################
######  check two signals edges
######################################################################
# context based to check if signal A toggle  before signal B or after in a time window
# @TODO not implemented yet


######################################################################
######  multiple signals status in a single timestamp
######################################################################
# timestamp value for a list of signals
# @TODO not implemented yet

if __name__ == "__main__":
    mcp.run()