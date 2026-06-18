"""Structured pre-processing for NgSpice simulation error logs.

Extracts deterministic facts from NgSpice output to provide structured
context for the LLM, reducing hallucination on weak local models.
"""

import re
from typing import Dict, List, Sequence, Tuple, Any

from chatbot.error_patterns import match_error_patterns, ErrorMatch
from chatbot.error_solutions import get_solution_for_category


# ── System prompt for error analysis ─────────────────────────────────────────

ERROR_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert circuit debugger inside eSim.\n\n"
    "TASK: Explain the pre-diagnosed error to the user in plain language.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- The error has ALREADY been diagnosed. Do NOT re-diagnose.\n"
    "- Use ONLY the [DETECTED ERROR] facts provided.\n"
    "- Focus on explaining WHY this error occurs in simple terms.\n"
    "- Reference the specific nodes/components mentioned.\n"
    "- Guide the user through the recommended fixes step-by-step.\n"
    "- Keep your explanation concise (max 150 words).\n\n"
    "OUTPUT FORMAT:\n"
    "1. **Error** — One-line summary\n"
    "2. **Why** — Brief root cause explanation\n"
    "3. **Fix** — Step-by-step using the provided eSim steps\n"
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


def rank_errors(matches: List[ErrorMatch]) -> List[Tuple[str, ErrorMatch]]:
    """Rank errors as 'Root Cause', 'Secondary Issue', or 'Consequence'.
    Returns list of (role_label, match) sorted by causal priority."""
    if not matches:
        return []
        
    sorted_matches = sorted(matches, key=lambda m: m.causal_priority)
    ranked = []
    
    # First item is root cause
    root_match = sorted_matches[0]
    ranked.append(("Root Cause", root_match))
    
    # Rest are secondary or consequence based on priority
    for m in sorted_matches[1:]:
        if m.causal_priority <= 2:
            ranked.append(("Secondary Issue", m))
        else:
            ranked.append(("Consequence", m))
            
    return ranked


def build_error_analysis_prompt(
    log_lines: Sequence[str],
    max_lines: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a structured prompt for the LLM from an NgSpice error log.

    Returns:
        A tuple of (prompt_string, list_of_tips_for_gui)
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
    error_line_count = 0
    for line in log_lines:
        if not any(p.search(line) for p in harmless_patterns):
            filtered_lines.append(line)
            if "error" in line.lower() or "failed" in line.lower():
                error_line_count += 1
            
    facts = extract_log_facts(filtered_lines)
    matches: List[ErrorMatch] = facts["error_patterns"]
    ranked_errors = rank_errors(matches)
    
    # Coverage tracking
    matched_lines = len(set(m.matched_text for m in matches))
    coverage = "High" if matched_lines >= error_line_count and error_line_count > 0 else ("Medium" if matched_lines > 0 else "Low")
    if error_line_count == 0 and matched_lines > 0:
        coverage = "High"

    sections = []

    # 1. Detected Error Block (Focus on Root Cause)
    tips = []
    if ranked_errors:
        root_role, root_match = ranked_errors[0]
        solution = get_solution_for_category(root_match.category)
        
        sections.append("[DETECTED ERROR]")
        sections.append(f"Error ID: {root_match.error_id}")
        sections.append(f"Category: {root_match.category}")
        sections.append(f"Role: {root_role}")
        sections.append(f"Diagnosis: {root_match.diagnosis}")
        
        for k, v in root_match.extracted_facts.items():
            sections.append(f"Detected {k}: {v}")
            
        sections.append("\nLikely Causes:")
        for cause in solution.get("likely_causes", []):
            sections.append(f"- {cause}")
            
        sections.append("\nRecommended Fixes:")
        for fix in solution.get("fixes", []):
            sections.append(f"- {fix}")
            
        sections.append("\neSim Steps:")
        for step in solution.get("esim_steps", []):
            sections.append(f"- {step}")
            
        if solution.get("prevention"):
            sections.append("\nPrevention:")
            for prev in solution.get("prevention", []):
                sections.append(f"- {prev}")
        sections.append("[END DETECTED ERROR]")
        
        # Prepare GUI tips (using the first fix as a tip)
        if solution.get("fixes"):
            tips.append({"fix": solution["fixes"][0]})
            
        # 2. Secondary Issues
        if len(ranked_errors) > 1:
            sections.append("\n[SECONDARY ISSUES]")
            for role, match in ranked_errors[1:]:
                sections.append(f"- {match.category} ({role})")
            sections.append("[END SECONDARY ISSUES]")

    # 3. Coverage tracking
    sections.append("\n[COVERAGE TRACKING]")
    sections.append(f"Detected Patterns: {len(matches)}")
    sections.append(f"Coverage: {coverage}")
    sections.append("[END COVERAGE TRACKING]")

    # 4. Circuit Context
    sections.append("\n[CIRCUIT CONTEXT]")
    if facts["circuit_name"]:
        sections.append(f"Circuit: {facts['circuit_name']}")
    if facts["simulation_type"]:
        sections.append(f"Simulation type: {facts['simulation_type']}")
    if facts["failed_nodes"]:
        sections.append(f"Mentioned nodes: {', '.join(facts['failed_nodes'])}")
    if facts["mentioned_components"]:
        sections.append(f"Mentioned components: {', '.join(facts['mentioned_components'])}")
    sections.append("[END CIRCUIT CONTEXT]")

    # 5. Raw Error Snippet Fallback
    sections.append("\n[RAW ERROR SNIPPET]")
    snippet_lines = filtered_lines[-max_lines:] if len(filtered_lines) > max_lines else filtered_lines
    sections.append("".join(snippet_lines).strip())
    sections.append("[END RAW ERROR SNIPPET]")

    return ("\n".join(sections), tips)
