"""Structured pre-processing for NgSpice simulation error logs.

Extracts deterministic facts from NgSpice output to provide structured
context for the LLM, reducing hallucination on weak local models.
"""

import re
from typing import Dict, List, Sequence, Tuple

from chatbot.error_patterns import match_error_patterns, format_error_context


# ── System prompt for error analysis ─────────────────────────────────────────

ERROR_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert circuit simulator debugger assistant.\n\n"
    "TASK: Analyze the provided NgSpice simulation error log and help the user understand the failure.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- Base your answer STRICTLY on the [SIMULATION ERROR LOG] and [DETECTED ERROR PATTERNS] provided below.\n"
    "- Ignore any mention of successful simulation times if the log text clearly shows a failure.\n\n"
    "OUTPUT FORMAT — use exactly these sections:\n"
    "1. **Error** — What error occurred\n"
    "2. **Cause** — Why this error happens in the circuit\n"
)


# ── Log fact extraction ──────────────────────────────────────────────────────

def extract_log_facts(log_lines: Sequence[str]) -> Dict[str, object]:
    """Extract structured facts from NgSpice log output."""
    full_text = "".join(log_lines)

    facts = {
        "total_lines": len(log_lines),
        "has_error": False,
        "error_patterns": [],
        "circuit_name": "",
        "simulation_type": "",
        "failed_nodes": [],
        "mentioned_components": [],
    }

    # Extract circuit name
    circuit_match = re.search(r"Circuit:\s*(.+)", full_text)
    if circuit_match:
        facts["circuit_name"] = circuit_match.group(1).strip()

    # Detect simulation type
    for sim_type in ("tran", "ac", "dc", "op", "noise"):
        if re.search(rf"\.{sim_type}\b", full_text, re.IGNORECASE):
            facts["simulation_type"] = sim_type
            break

    # Check for errors
    error_indicators = [
        "error", "failed", "singular", "convergence", "abort",
        "cannot", "not found", "undefined", "too small",
    ]
    facts["has_error"] = any(
        indicator in full_text.lower() for indicator in error_indicators
    )

    # Match known patterns
    facts["error_patterns"] = match_error_patterns(full_text)

    # Extract mentioned node names
    node_matches = re.findall(r"node\s+['\"]?(\S+)['\"]?", full_text, re.IGNORECASE)
    facts["failed_nodes"] = list(dict.fromkeys(node_matches))[:10]

    # Extract mentioned component references
    comp_matches = re.findall(
        r"\b([RCLVIDQMX]\d+)\b", full_text, re.IGNORECASE
    )
    facts["mentioned_components"] = list(dict.fromkeys(comp_matches))[:20]

    return facts


def build_error_analysis_prompt(
    log_lines: Sequence[str],
    max_lines: int = 60,
) -> Tuple[str, List[Dict[str, str]]]:
    """Build a structured prompt for the LLM from an NgSpice error log.

    Returns:
        A tuple of (prompt_string, list_of_matched_patterns)
    """
    # Filter out harmless eSim default model warnings to prevent LLM hallucinations
    harmless_patterns = [
        re.compile(r"unable to find definition of model esim_", re.IGNORECASE),
        re.compile(r"-\s*default assumed", re.IGNORECASE),
        re.compile(r"^\*\* ngspice-\d+", re.IGNORECASE),
        re.compile(r"^\*\* The U\. C\. Berkeley CAD Group", re.IGNORECASE),
        re.compile(r"^\*\* Copyright", re.IGNORECASE),
        re.compile(r"^\*\* Please get your ngspice manual", re.IGNORECASE),
        re.compile(r"^\*\* Please file your bug-reports", re.IGNORECASE),
        re.compile(r"^\*{6,}$"),  # Matches the ****** banner separators
    ]
    
    filtered_lines = []
    for line in log_lines:
        if not any(p.search(line) for p in harmless_patterns):
            filtered_lines.append(line)
            
    log_lines = filtered_lines
    facts = extract_log_facts(log_lines)

    # Truncate log to last N lines (errors are typically at the end)
    if len(log_lines) > max_lines:
        truncated = True
        display_lines = log_lines[-max_lines:]
    else:
        truncated = False
        display_lines = log_lines

    log_text = "".join(display_lines).strip()
    pattern_context = format_error_context(facts["error_patterns"])

    sections = []

    if pattern_context:
        sections.append(pattern_context)

    sections.append("[LOG FACTS]")
    if facts["circuit_name"]:
        sections.append(f"Circuit: {facts['circuit_name']}")
    if facts["simulation_type"]:
        sections.append(f"Simulation type: {facts['simulation_type']}")
    sections.append(f"Total log lines: {facts['total_lines']}")
    if truncated:
        sections.append(f"Showing last {max_lines} lines")
    if facts["failed_nodes"]:
        sections.append(f"Mentioned nodes: {', '.join(facts['failed_nodes'])}")
    if facts["mentioned_components"]:
        sections.append(f"Mentioned components: {', '.join(facts['mentioned_components'])}")
    sections.append("[END LOG FACTS]")

    sections.append("")
    sections.append("[SIMULATION ERROR LOG]")
    sections.append(log_text)
    sections.append("[END SIMULATION ERROR LOG]")

    return ("\n".join(sections), facts["error_patterns"])
