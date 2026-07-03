"""Offline (LLM-free) response formatters for the eSim AI Assistant.

When Ollama is unavailable, these functions format the deterministic
pipeline output into structured plain-text responses.  All diagnostic
data comes from the existing error and netlist analysis pipelines;
no LLM call is made.
"""

import re
from typing import Dict, List, Sequence, Tuple, Any

from chatbot.error_log_analysis import (
    extract_log_facts,
    rank_errors,
    _has_explicit_file_error,
    _filter_likely_causes,
)
from chatbot.error_solutions import get_solution_for_category
from chatbot.netlist_analysis import (
    ParsedNetlist,
    detect_circuit_blocks,
    format_netlist_table,
)


# ── Harmless log patterns (same as in build_error_analysis_prompt) ────────────
# These are filtered out before analysis to prevent false positives.

_HARMLESS_PATTERNS = [
    re.compile(r"unable to find definition of model esim_", re.IGNORECASE),
    re.compile(r"-\s*default assumed", re.IGNORECASE),
    re.compile(r"^\*\* ngspice-\d+", re.IGNORECASE),
    re.compile(r"^\*\* The U\. C\. Berkeley CAD Group", re.IGNORECASE),
    re.compile(r"^\*\* Copyright", re.IGNORECASE),
    re.compile(r"^\*\* Please get your ngspice manual", re.IGNORECASE),
    re.compile(r"^\*\* Please file your bug-reports", re.IGNORECASE),
    re.compile(r"^\*{6,}$"),
]

_OFFLINE_FOOTNOTE = (
    "\n\n---\n"
    "💡 *Note: This analysis was generated using the deterministic "
    "pipeline without an LLM. For a more detailed natural-language "
    "explanation, start **Ollama** with a local model and retry.*"
)


# ── Error analysis offline formatter ─────────────────────────────────────────

def format_error_analysis_offline(
    log_lines: Sequence[str],
) -> str:
    """Format error analysis results as structured text without an LLM.

    Runs the full deterministic error pipeline (regex matching, ranking,
    suppression, solution lookup) and formats the output directly.

    Returns:
        A plain-text response string ready to display in the chat UI.
    """
    # Filter out harmless NgSpice banners and default-model warnings
    filtered_lines: List[str] = [
        line for line in log_lines
        if not any(p.search(line) for p in _HARMLESS_PATTERNS)
    ]

    if not filtered_lines:
        return "⚠️ **No meaningful error output found in the log.**" + _OFFLINE_FOOTNOTE

    facts = extract_log_facts(filtered_lines)
    matches = facts["error_patterns"]
    ranked_errors = rank_errors(matches)

    if not ranked_errors:
        return (
            "⚠️ **No known error patterns were detected in the log.**\n\n"
            "*The simulation output may contain warnings or non-standard "
            "messages that the deterministic pipeline does not yet cover.*"
            + _OFFLINE_FOOTNOTE
        )

    full_text = "".join(filtered_lines)
    has_file_error = _has_explicit_file_error(full_text)

    root_causes = [r for r in ranked_errors if r.role == "Root Cause"]
    non_roots = [r for r in ranked_errors if r.role != "Root Cause"]
    total_errors = len(ranked_errors)

    sections: List[str] = []
    sections.append("### 🛠️ Error Analysis Results\n")

    # ── Root causes ──────────────────────────────────────────────────
    for idx, rc in enumerate(root_causes, 1):
        m = rc.match
        solution = get_solution_for_category(m.category)
        likely_causes = _filter_likely_causes(
            solution.get("likely_causes", []), has_file_error
        )

        role_label = "Root Cause" if rc.role == "Root Cause" else rc.role
        if total_errors > 1:
            sections.append(
                f"#### 🔴 Error {idx} of {total_errors}: **{m.category}** `({role_label})`"
            )
        else:
            sections.append(f"#### 🔴 Detected Error: **{m.category}** `({role_label})`")

        severity = solution.get('severity', 'unknown').upper()
        sections.append(f"- **Severity:** `{severity}`")
        sections.append("")
        sections.append(f"**📋 Diagnosis:**\n{m.diagnosis}")

        # Entity extraction (node name, model name, etc.)
        if m.extracted_facts:
            sections.append("")
            for key, value in m.extracted_facts.items():
                sections.append(f"- **Detected {key.title()}:** `{value}`")

        if likely_causes:
            sections.append("\n**❓ Likely Causes:**")
            for cause in likely_causes:
                sections.append(f"- {cause}")

        fixes = solution.get("fixes", [])
        if fixes:
            sections.append("\n**🔧 Recommended Fixes:**")
            for i, fix in enumerate(fixes, 1):
                sections.append(f"{i}. **{fix}**")

        esim_steps = solution.get("esim_steps", [])
        if esim_steps:
            sections.append("\n**📝 eSim Steps:**")
            for i, step in enumerate(esim_steps, 1):
                sections.append(f"{i}. {step}")

        prevention = solution.get("prevention", [])
        if prevention:
            sections.append("\n**🛡️ Prevention:**")
            for tip in prevention:
                sections.append(f"- *{tip}*")

        sections.append("")

    # ── Secondary / consequence issues ───────────────────────────────
    if non_roots:
        sections.append("#### ⚠️ Secondary Issues")
        for r in non_roots:
            label = f"**{r.match.category}** `({r.role})`"
            if r.suppressed_by:
                label += f" *(related to `{r.suppressed_by}`)*"
            solution = get_solution_for_category(r.match.category)
            fix = solution.get("fixes", [""])[0]
            sections.append(f"- {label}")
            if fix:
                sections.append(f"  - *Fix:* {fix}")
        sections.append("")

    # ── Circuit context ──────────────────────────────────────────────
    context_parts = []
    if facts.get("circuit_name"):
        context_parts.append(f"- **Circuit:** `{facts['circuit_name']}`")
    if facts.get("simulation_type"):
        context_parts.append(f"- **Simulation Type:** `{facts['simulation_type']}`")
    if facts.get("failed_nodes"):
        nodes_str = ", ".join(f"`{n}`" for n in facts['failed_nodes'])
        context_parts.append(f"- **Mentioned Nodes:** {nodes_str}")
    if facts.get("mentioned_components"):
        comps_str = ", ".join(f"`{c}`" for c in facts['mentioned_components'])
        context_parts.append(f"- **Mentioned Components:** {comps_str}")
    if context_parts:
        sections.append("#### 🔌 Circuit Context")
        sections.extend(context_parts)
        sections.append("")

    sections.append(_OFFLINE_FOOTNOTE)

    return "\n".join(sections)


# ── Netlist analysis offline formatter ───────────────────────────────────────

def format_netlist_analysis_offline(
    parsed: ParsedNetlist,
    raw_lines: Sequence[str],
) -> str:
    """Format netlist analysis results as structured text without an LLM.

    Uses the existing ``format_netlist_table()`` for component and simulation
    setup formatting, then appends obvious issues and detected circuit blocks.

    Returns:
        A Markdown-formatted response string ready to display in the chat UI.
    """
    sections: List[str] = []

    # ── Component table and simulation setup (already implemented) ────
    table_md = format_netlist_table(parsed)
    sections.append(table_md)

    # ── Obvious issues ───────────────────────────────────────────────
    issues: List[str] = []
    if not parsed.reference_node_0_present and not parsed.gnd_label_present:
        issues.append("Missing reference ground (node `0` or `GND`).")
    if not parsed.analysis_directives:
        issues.append(
            "No simulation directives (e.g., `.tran`, `.dc`, `.ac`) found."
        )
    if parsed.unresolved_subckt_calls:
        names = ", ".join(
            f"`{c.subcircuit}`" for c in parsed.unresolved_subckt_calls
        )
        issues.append(f"Unresolved subcircuits: {names}")
    missing_includes = [
        f"`{item.token}`" for item in parsed.includes if not item.exists
    ]
    if missing_includes:
        issues.append(
            f"Missing included files: {', '.join(missing_includes)}"
        )

    if issues:
        sections.append("\n### ⚠️ Issues Found")
        for issue in issues:
            sections.append(f"- **{issue}**")
    else:
        sections.append("\n### ✅ Issues Found")
        sections.append("- *No obvious issues detected.*")

    # ── Detected circuit blocks ──────────────────────────────────────
    blocks = detect_circuit_blocks(parsed)
    high_conf = [b for b in blocks if b[1] == "HIGH"]

    if high_conf:
        sections.append("\n### 🧩 Detected Circuit Blocks")
        for block_name, _, relationship in high_conf:
            sections.append(f"- **{block_name}** — *{relationship}*")
    else:
        sections.append("\n### 🧩 Detected Circuit Blocks")
        sections.append(
            "- *No standard circuit blocks detected with high confidence.*"
        )

    # ── Model and subcircuit definitions ─────────────────────────────
    if parsed.model_names:
        models_str = ", ".join(f"`{m}`" for m in parsed.model_names)
        sections.append(
            f"\n**Defined Models:** {models_str}"
        )
    if parsed.subckt_definitions:
        subckts_str = ", ".join(f"`{s}`" for s in parsed.subckt_definitions)
        sections.append(
            f"**Defined Subcircuits:** {subckts_str}"
        )

    sections.append(_OFFLINE_FOOTNOTE)

    return "\n".join(sections)
