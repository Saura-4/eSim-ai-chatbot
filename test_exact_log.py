import sys
import json
sys.path.insert(0, "src")
from chatbot.error_log_analysis import build_error_analysis_prompt, extract_log_facts

log_text = """warning, can't find model 'esim_diode' from line
    d1 in2 net-_c1-pad1_ esim_diode
warning, can't find model 'esim_diode' from line
    d2 gnd in2 esim_diode
warning, can't find model 'esim_diode' from line
    d4 gnd in1 esim_diode
warning, can't find model 'esim_diode' from line
    d3 in1 net-_c1-pad1_ esim_diode
warning, can't find model 'esim_diode' from line
    d5 gnd out esim_diode

Note: No compatibility mode selected!


Circuit: * /home/saurabh/pilot_related/new_installer/esim-1.1.2/examples/7805voltageregulator/7805voltageregulator.cir

Error on line 4 or its substitute:
  d1 in2 net-_c1-pad1_ esim_diode
could not find a valid modelname
    Simulation interrupted due to error!

Error: circuit not parsed.
Note: no resource usage information for 'time',
	or no active circuit available


Total elapsed time (seconds) = 0.002 

Total DRAM available = 3398.160 MB.
DRAM currently available =  437.016 MB.
Maximum ngspice program size =   22.309 MB.
Current ngspice program size =   14.055 MB.

Shared ngspice pages =   12.090 MB.
Text (code) pages =    6.406 MB.
Stack = 0 bytes.
Library pages =    2.105 MB.


Simulation Failed!"""

lines = [ln+"\n" for ln in log_text.split("\n")]

# 1. Test the filter logic
import re
harmless_patterns = [
    re.compile(r"unable to find definition of model esim_", re.IGNORECASE),
    re.compile(r"-\s*default assumed", re.IGNORECASE),
]

filtered_lines = []
for line in lines:
    if not any(p.search(line) for p in harmless_patterns):
        filtered_lines.append(line)

print("--- After Filter ---")
print("Filtered count:", len(filtered_lines), "Original:", len(lines))

# 2. Test fact extraction
facts = extract_log_facts(filtered_lines)
print("\n--- Extracted Facts ---")
print(json.dumps(facts, indent=2, default=str))

# 3. Test prompt generation
print("\n--- Final Prompt ---")
print(build_error_analysis_prompt(lines))
