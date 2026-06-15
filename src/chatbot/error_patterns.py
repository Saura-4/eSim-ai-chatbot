"""Deterministic pattern matching for common NgSpice simulation errors.

This module matches known error patterns in NgSpice log output and provides
structured diagnoses + fix suggestions. Results are included in the LLM
prompt so even weak local models can give accurate advice.
"""

import re
from typing import Dict, List, Tuple


# Each pattern: (compiled_regex, error_category, diagnosis, fix_suggestion)
_ERROR_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"singular matrix", re.IGNORECASE),
        "Singular Matrix",
        "The circuit matrix is singular — NgSpice cannot solve the DC operating point.",
        "Add `.options gmin=1e-12 reltol=0.01` to the netlist. "
        "Check for floating nodes or missing ground connections.",
    ),
    (
        re.compile(r"timestep too small", re.IGNORECASE),
        "Timestep Too Small",
        "Transient analysis failed because the simulator could not converge within the minimum timestep.",
        "Try: increase TSTEP in .tran, add `.options method=gear`, "
        "or add `.options reltol=0.01 abstol=1e-9`.",
    ),
    (
        re.compile(r"no dc path to ground", re.IGNORECASE),
        "No DC Path to Ground",
        "A node has no DC path to ground (node 0). Every node must have a resistive path to ground.",
        "Add a high-value resistor (e.g. 1G ohm) from the floating node to ground (node 0).",
    ),
    (
        re.compile(r"(?:(?:model|device)\s+['\"]?(\S+)['\"]?\s+(?:not found|undefined|unknown)|can'?t\s+find\s+model|could\s+not\s+find\s+a\s+valid\s+modelname)", re.IGNORECASE),
        "Missing Model",
        "A component references a model/device that is not defined in the netlist.",
        "Add the missing .model statement or .include the library file "
        "that defines this model.",
    ),
    (
        re.compile(r"can'?t\s+find\s+init\s+file", re.IGNORECASE),
        "Init File Missing",
        "NgSpice cannot find its initialization file (.spiceinit or spinit).",
        "Check NgSpice installation. The init file should be in the NgSpice share directory.",
    ),
    (
        re.compile(r"doAnalyses:\s+TRAN\s+?.*failed", re.IGNORECASE),
        "Transient Analysis Failed",
        "The transient (.tran) analysis did not complete successfully.",
        "Check simulation parameters. Try reducing TSTOP or increasing TSTEP. "
        "Also verify all component values are reasonable.",
    ),
    (
        re.compile(r"(?:non-?convergence|failed\s+to\s+converge|convergence\s+fail)", re.IGNORECASE),
        "Convergence Failure",
        "The simulator could not converge to a solution.",
        "Try: `.options reltol=0.01 abstol=1e-9 gmin=1e-12 itl1=200 itl4=50`. "
        "Also check for unrealistic component values.",
    ),
    (
        re.compile(r"too\s+many\s+iterations", re.IGNORECASE),
        "Too Many Iterations",
        "The DC operating point or transient step needed more iterations than allowed.",
        "Increase iteration limits with `.options itl1=300 itl4=100`. "
        "If problem persists, simplify the circuit or check for positive feedback loops.",
    ),
    (
        re.compile(r"(?:voltage|current)\s+source\s+loop", re.IGNORECASE),
        "Source Loop",
        "Voltage sources form a loop, or current sources feed each other without a path.",
        "Add a small series resistor (e.g. 1m ohm) to break the voltage source loop, "
        "or verify the source topology.",
    ),
    (
        re.compile(r"subcircuit\s+['\"]?(\S+)['\"]?\s+not\s+found", re.IGNORECASE),
        "Missing Subcircuit",
        "A subcircuit instantiation (X component) references a .subckt that is not defined.",
        "Add the missing .subckt definition or .include the file containing it.",
    ),
]

# Patterns to extract the specific node/component/model name from the error
_NODE_PATTERN = re.compile(r"(?:node|net)\s+['\"]?(\S+)['\"]?", re.IGNORECASE)
_MODEL_PATTERN = re.compile(
    r"(?:model|device|subcircuit)\s+['\"]?(\S+)['\"]?", re.IGNORECASE
)


def match_error_patterns(log_text: str) -> List[Dict[str, str]]:
    """Match known NgSpice error patterns in the log text.

    Returns a list of dicts, each with keys:
      category, diagnosis, fix, matched_text
    """
    matches = []
    seen_categories = set()

    for pattern, category, diagnosis, fix in _ERROR_PATTERNS:
        m = pattern.search(log_text)
        if m and category not in seen_categories:
            seen_categories.add(category)
            matched = m.group(0)

            # Try to extract a specific entity name
            entity = ""
            if category in ("Missing Model", "Missing Subcircuit"):
                entity_match = _MODEL_PATTERN.search(matched)
                if entity_match:
                    entity = entity_match.group(1)
            elif category == "No DC Path to Ground":
                node_match = _NODE_PATTERN.search(log_text[max(0, m.start()-80):m.end()+80])
                if node_match:
                    entity = node_match.group(1)

            result = {
                "category": category,
                "diagnosis": diagnosis,
                "fix": fix,
                "matched_text": matched,
            }
            if entity:
                result["entity"] = entity
            matches.append(result)

    return matches


def format_error_context(matches: List[Dict[str, str]]) -> str:
    """Format matched error patterns into a structured context block for the LLM.

    This block is prepended to the error log so the LLM has deterministic
    information about what went wrong, reducing hallucination.
    """
    if not matches:
        return ""

    lines = ["[DETECTED ERROR PATTERNS]"]
    for i, m in enumerate(matches, 1):
        lines.append(f"Error {i}: {m['category']}")
        lines.append(f"  Diagnosis: {m['diagnosis']}")
        lines.append(f"  Suggested fix: {m['fix']}")
        if "entity" in m:
            lines.append(f"  Related entity: {m['entity']}")
        lines.append("")
    lines.append("[END DETECTED ERROR PATTERNS]")
    return "\n".join(lines)
