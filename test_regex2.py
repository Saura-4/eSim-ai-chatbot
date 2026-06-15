import re

text = """d1 in2 net-_c1-pad1_ esim_diode
could not find a valid modelname
Simulation interrupted due to error!"""

pat = re.compile(r"(?:(?:model|device)\s+['\"]?(\S+)['\"]?\s+(?:not found|undefined|unknown)|can'?t\s+find\s+model|could\s+not\s+find\s+a\s+valid\s+modelname)", re.IGNORECASE)

print(pat.search(text))
