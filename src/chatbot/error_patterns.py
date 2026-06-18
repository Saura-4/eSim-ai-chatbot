"""Deterministic pattern matching for common NgSpice simulation errors.

This module matches known error patterns in NgSpice log output and provides
structured diagnoses + fix suggestions. Results are included in the LLM
prompt so even weak local models can give accurate advice.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Any

@dataclass
class ErrorMatch:
    error_id: str
    category: str
    causal_priority: int      # 1 = root cause, 2 = intermediate, 3 = consequence
    diagnosis: str
    matched_text: str
    extracted_facts: Dict[str, str] = field(default_factory=dict)

# Patterns to extract specific entities
_NODE_PATTERN = re.compile(r"(?:node|net)\s+['\"]?([^'\"\s]+)['\"]?", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"(?:model|device|subcircuit)\s+['\"]?([^'\"\s]+)['\"]?", re.IGNORECASE)
_SOURCE_PATTERN = re.compile(r"source\s+([VvIi][^\s]+)", re.IGNORECASE)

def _extract_model_facts(log_text: str, match_str: str) -> Dict[str, str]:
    facts = {}
    m = _MODEL_PATTERN.search(match_str)
    if m:
        facts["Model Name"] = m.group(1)
    return facts

def _extract_subckt_facts(log_text: str, match_str: str) -> Dict[str, str]:
    facts = {}
    m = _MODEL_PATTERN.search(match_str)
    if m:
        facts["Subcircuit Name"] = m.group(1)
    return facts

def _extract_node_facts(log_text: str, match_str: str, match_pos: int) -> Dict[str, str]:
    facts = {}
    # Look around the match position
    context = log_text[max(0, match_pos - 80):min(len(log_text), match_pos + 80)]
    m = _NODE_PATTERN.search(context)
    if m:
        facts["Node"] = m.group(1)
    return facts

def _extract_source_loop_facts(log_text: str, match_str: str) -> Dict[str, str]:
    facts = {}
    m = _SOURCE_PATTERN.search(match_str)
    if m:
        facts["Source"] = m.group(1)
    return facts

# Each pattern: (compiled_regex, error_id, category, causal_priority, diagnosis, extractor)
_ERROR_PATTERNS: List[Tuple[re.Pattern, str, str, int, str, Any]] = [
    (
        re.compile(r"singular matrix", re.IGNORECASE),
        "ERR004", "Singular Matrix", 2,
        "The circuit matrix is singular — NgSpice cannot solve the DC operating point.",
        None
    ),
    (
        re.compile(r"timestep too small", re.IGNORECASE),
        "ERR005", "Timestep Too Small", 2,
        "Transient analysis failed because the simulator could not converge within the minimum timestep.",
        None
    ),
    (
        re.compile(r"no dc path to ground", re.IGNORECASE),
        "ERR003", "No DC Path to Ground", 1,
        "A node has no DC path to ground (node 0). Every node must have a resistive path to ground.",
        _extract_node_facts
    ),
    (
        re.compile(r"(?:(?:model|device)\s+['\"]?(\S+)['\"]?\s+(?:not found|undefined|unknown)|can'?t\s+find\s+model|could\s+not\s+find\s+a\s+valid\s+modelname)", re.IGNORECASE),
        "ERR001", "Missing Model", 1,
        "A component references a model/device that is not defined in the netlist.",
        _extract_model_facts
    ),
    (
        re.compile(r"Unknown\s+model\s+type\s+(\S+)\s+-\s+ignored", re.IGNORECASE),
        "ERR006", "Invalid Model Syntax", 1,
        "A .model statement has an invalid type or incorrect parameter ordering. The type must immediately follow the model name.",
        _extract_model_facts
    ),
    (
        re.compile(r"can'?t\s+find\s+init\s+file", re.IGNORECASE),
        "ERR007", "Init File Missing", 1,
        "NgSpice cannot find its initialization file (.spiceinit or spinit).",
        None
    ),
    (
        re.compile(r"doAnalyses:\s+TRAN\s+?.*failed", re.IGNORECASE),
        "ERR008", "Transient Analysis Failed", 3,
        "The transient (.tran) analysis did not complete successfully.",
        None
    ),
    (
        re.compile(r"(?:non-?convergence|failed\s+to\s+converge|convergence\s+fail)", re.IGNORECASE),
        "ERR009", "Convergence Failure", 2,
        "The simulator could not converge to a solution.",
        None
    ),
    (
        re.compile(r"too\s+many\s+iterations", re.IGNORECASE),
        "ERR010", "Too Many Iterations", 2,
        "The DC operating point or transient step needed more iterations than allowed.",
        None
    ),
    (
        re.compile(r"(?:voltage|current)\s+source\s+loop|singular\s+matrix.*?#branch", re.IGNORECASE),
        "ERR011", "Source Loop", 1,
        "Voltage sources form a loop, or current sources feed each other without a path.",
        _extract_source_loop_facts
    ),
    (
        re.compile(r"subcircuit\s+['\"]?(\S+)['\"]?\s+not\s+found", re.IGNORECASE),
        "ERR002", "Missing Subcircuit", 1,
        "A subcircuit instantiation (X component) references a .subckt that is not defined.",
        _extract_subckt_facts
    ),
    (
        re.compile(r"(?:no\s+such\s+parameter|parameter\s+is\s+missing|unknown\s+parameter)", re.IGNORECASE),
        "ERR012", "Invalid Parameter / Syntax Error", 1,
        "A component has an invalid parameter or is missing required parameters.",
        None
    ),
    (
        re.compile(r"Simulation Completed Successfully!", re.IGNORECASE),
        "ERR013", "No Plot Data (Simulation Succeeded)", 0,
        "The simulation actually completed successfully with no errors, but there is no data to plot.",
        None
    ),
]


def match_error_patterns(log_text: str) -> List[ErrorMatch]:
    """Match known NgSpice error patterns in the log text.

    Returns a list of ErrorMatch objects.
    """
    matches = []
    seen_categories = set()

    for pattern, error_id, category, causal_priority, diagnosis, extractor in _ERROR_PATTERNS:
        m = pattern.search(log_text)
        if m and category not in seen_categories:
            seen_categories.add(category)
            matched = m.group(0)

            facts = {}
            if extractor:
                if extractor == _extract_node_facts:
                    facts = extractor(log_text, matched, m.start())
                else:
                    facts = extractor(log_text, matched)

            matches.append(ErrorMatch(
                error_id=error_id,
                category=category,
                causal_priority=causal_priority,
                diagnosis=diagnosis,
                matched_text=matched,
                extracted_facts=facts
            ))

    return matches
