Root cause analysis Debug Agent for RTL verification 

This is an an agent project for RTL verification debug, it is built on MCP and openAI's GPT-4o
the main purpose of this agent is to parse a log file based on a UVM/Systemverilog simulation and try to debug the found errors
the prompt entry is that this agent is like a verification engineer that will try to use different tools to explore the environement and root cause the failure 

It uses the Reason+Act loop to call the log files, the source code and the vcd waveform to debug the errors 

It can open verilog/systemverilog files, vcd waveform files and log files to extract simulation details and information
Due to the possibility of having huge files to parse, multiple python functions are used to do the parsing in small chunks that the agent could handle 

Performed Main actions
waveform analysis, open a vcd file and extract different values of a different signals and transitions using multiple functions 
log files : open log files and extract related source code files and paths and simulation messages 
source code files : open verilog and systemverilog files to identify the context of different signals and what is the signal to probe 

Project Structure
engine_test.py: The Agent Client. Contains the OpenAI loop and logic.
server_test.py: The MCP Server. Hosts the Python tools for file and VCD manipulation.
sim.log / top.sv / wave.vcd: some examples from a simulation to test the code

Setup & Installation
Clone the Repo:Bashgit clone https://github.com/slimmsehli/RCAnalysis.git
cd RCAnalysis
Install Dependencies: Bashpip install mcp openai vcdvcd
OPENAI_API_KEY is set to be read from .env file so make sure that you add it to your .env and it is already removed from git 

Current status :
it can perform 15 iteration to extract a UVM erros from a log file, search the related source code and check the signal in question from the vcd 
multiple functions need to be added to this inorder to enahnce the agent, feel free to add what you can see fits 


######### Currently implmented functions for VCD parsing

######### General functions for vcd details 
1. vcd_get_simulation_time(vcd_path : str, store_scopes: bool = False) -> float : 
	--> get the full simulation time as float
2. vcd_get_timescale_scal(path: str, store_scopes: bool = False) -> List[Tuple[Any, Any]]  :
	--> get the simulation timescale, the return are a tuple the magnitude and the unit eg: 1, ns
3. list_vcd_signals(path: str, pattern: str = "", store_scopes: bool = False) -> str: 
	--> get the full list of the signals in the vcd file

######### Functions for probing signals from the vcd
1. vcd_get_signal_value_at_timestamp(path: str, signal_name: str, timestamp: Union[str, float, int], method: str = "previous") -> Any:
	--> get the value a signal at a specific timestamp, the return is a int
2. vcd_get_signal_values_in_timeframe(path: str, signal_name: str, start: Optional[Union[str, float, int]], end: Optional[Union[str, float, int]], include_start_prev: bool = True) -> List[Tuple[float, Any]]:
--> get all the values of a signal inside a time window in simulation, the return is a table of all the values and the timestamp
3. vcd_count_signal_all_transitions(path: str, signal_name: str, edge: str, start: Optional[Union[str, float, int]], end: Optional[Union[str, float, int]], bit_index: Optional[int] = None) -> int:
	--> get the number of a signal transition inside a timewindow, the rutrn is an int and the edge could be specified for rising or falling, this does not work on bus
4. vcd_next_change_after(path: str, signal_name: str, timestamp: Union[str, float, int]) -> Optional[Tuple[float, Any]]:
	--> find the next signal transition after a specific timestamp
5. vcd_prev_change_before(path: str, signal_name: str, timestamp: Union[str, float, int]) -> Optional[Tuple[float, Any]]:
	--> find the previous signal transition before a specific timestamp
6. vcd_search_value(path: str, signal_name: str, value: Any, start: Optional[Union[str, float, int]] = None, end: Optional[Union[str, float, int]] = None) -> List[float]:
	--> search in the vcd if a signal has assigned to a specific vlaue in a time window
7. vcd_get_signals_values_at_timestamp(path: str, signal_names: Iterable[str], timestamp: Union[str, float, int], method: str = "previous") -> Dict[str, Any]:
	--> for a list of signals, return the value of each one of them at a specific timestamp
8. vcd_get_signals_aligned_in_window(path: str, signal_names: Iterable[str], start: Union[str, float, int], end: Union[str, float, int]) -> Tuple[List[float], Dict[str, List[Any]]]:
	--> for a list of signals given, return all the transition values of all signals with their timestamp all signals are aligned on the smae timeline

######### Functions for parsing log files and source code files
1. parse_log_for_errors(log_path: str, error_regx : str = r"ERROR:\s+([\d\w\.\/]+)\s+at\s+line\s+(\d+)") -> str:
	--> search for error patterns inside a log file
2. find_first_uvm_error(log_path: str) -> List[Tuple[Any, Any]]:
	--> find the first error in a log file and return the error line at its line number 
3. get_error_context(log_path: str, error_line: int, window: int = 20) -> str:
	--> extract the N lines before an error line, to undrestand more the context of the error
4. search_log_keyword(log_path: str, keyword: str, context_lines: int = 10) -> str:
	--> search for a specific keyword in a logfile and return the surounding N lines of the keyword line
5. get_source_snippet(file_path: str, line_number: int, context: int = 5) -> str:
	--> get a code snipet of a source code file bsed opn line number, this will extract the previous and the after N lines from the given line number

######### Not yet implemented functions

######### Not yet implemented functions for vcd details navigation 
vcd_list_scopes(path: str) -> List[str]:
	--> Return hierarchical scopes/modules captured in the vcd
vcd_get_signal_info(path: str, signal_name: str) -> Dict[str, Any]:
	--> Return bit-width, scope, identifier code, is_bus, original declaration name
vcd_resolve_hierarchical(path: str, partial_name: str) -> List[str]:
	--> Resolve fuzzy/partial names to full hierarchical signal paths
vcd_list_bus_members(path: str, bus_name: str) -> List[str]:
	--> List bit members (e.g., data[31], data[30], ...) for a bus
vcd_timescale_to_seconds(magnitude: int, unit: str) -> float:
	--> Convert timescale tuple (e.g., 1, 'ns') to seconds multiplier

######### Not yet implemented functions for vcd details navigation 
vcd_detect_glitches(path: str, signal_name: str, min_pulse_width: float, start: Optional[float], end: Optional[float]) -> List[Tuple[float, float]]:
	--> Find pulses narrower than min_pulse_width; return (t_rise, t_fall)
vcd_track_xz(path: str, signal_name: str, start: Optional[float], end: Optional[float]) -> List[Tuple[float, Any]]:
	--> Return timestamps where value is 'x' or 'z'; for buses include vector
vcd_jump_to_nth_transition(path: str, signal_name: str, n: int, start: Optional[float]=None, end: Optional[float]=None) -> Optional[Tuple[float, Any]]:
	--> Navigate directly to nth transition in window
vcd_align_to_clock_edges(path: str, signal_names: Iterable[str], clock: str, start: float, end: float, edge: str="rising") -> Tuple[List[float], Dict[str, List[Any]]]:
	--> Sample listed signals at each clock edge; useful for cycle-accurate comparisons

######### Not yet implemented functions log file parsing
uvm_count_severity(log_path: str) -> Dict[str, int]:
	--> Count UVM_INFO/WARNING/ERROR/FATAL across the log
uvm_extract_test_seed(log_path: str) -> Dict[str, Any]:
	--> Extract test name, random seed, pluses (+UVM_TESTNAME, +ntb_random_seed, etc.)
uvm_phase_timeline(log_path: str) -> List[Tuple[str, Optional[float]]]:
	--> Parse phase enter/exit and timestamps (if present)
uvm_find_first_fatal(log_path: str) -> Optional[Tuple[int, str]]:
	--> Return (line_number, line_text) for first UVM_FATAL
log_split_by_section(log_path: str, markers: List[str]) -> Dict[str, List[str]]:
	--> Split file by markers (e.g., compile/elab/run) for targeted parsing
log_extract_stacktrace(log_path: str, start_line: int) -> List[str]:
	--> Collect stack trace following an error line (simulator-specific formatting)
log_detect_simulator(log_path: str) -> str:
	--> Return simulator inferred from banner (e.g., VCS, Questa, Xcelium)
log_extract_timestamps(log_path: str, pattern: str=r"\[(\d+:\d+:\d+)\]") -> List[Tuple[int, str]]:
	--> Return (line_number, hh:mm:ss) pairs; useful to correlate with waveform time bases







