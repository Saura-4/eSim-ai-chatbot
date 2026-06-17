"""Deterministic helpers for grounding AI netlist analysis.

This module intentionally performs lightweight SPICE fact extraction only. It
does not simulate the circuit or try to prove electrical correctness.
"""

from dataclasses import dataclass
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple


MAX_NETLIST_CONTEXT_CHARS = 7000
MAX_NETLIST_FACT_ITEMS = 40

_COMPONENT_PREFIXES = set("RCLVIDQMEFGHJKTUWXZ")
_ANALYSIS_DIRECTIVES = {
    ".op", ".dc", ".ac", ".tran", ".noise", ".tf", ".pz", ".sens",
}
_OUTPUT_DIRECTIVES = {
    ".plot", ".print", ".probe", ".save", ".meas", ".measure",
}
_CONTROL_OUTPUT_COMMANDS = {
    "plot", "print", "wrdata", "write", "save", "meas", "measure",
}


@dataclass(frozen=True)
class ComponentFact:
    reference: str
    prefix: str
    nodes: Tuple[str, ...]
    value_or_model: str
    raw: str


@dataclass(frozen=True)
class IncludeFact:
    token: str
    resolved_path: str
    exists: bool


@dataclass(frozen=True)
class SubcircuitCallFact:
    reference: str
    subcircuit: str
    nodes: Tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class ParsedNetlist:
    filename: str
    path: str
    total_lines: int
    active_lines: Tuple[str, ...]
    comment_lines: Tuple[str, ...]
    ignored_comment_component_like_lines: Tuple[str, ...]
    components: Tuple[ComponentFact, ...]
    nodes: Tuple[str, ...]
    ordinary_directives: Tuple[str, ...]
    analysis_directives: Tuple[str, ...]
    control_block_lines: Tuple[str, ...]
    control_commands: Tuple[str, ...]
    output_commands: Tuple[str, ...]
    includes: Tuple[IncludeFact, ...]
    model_names: Tuple[str, ...]
    subckt_definitions: Tuple[str, ...]
    subckt_calls: Tuple[SubcircuitCallFact, ...]
    unresolved_subckt_calls: Tuple[SubcircuitCallFact, ...]
    voltage_sources: Tuple[str, ...]
    load_candidates: Tuple[str, ...]
    reference_node_0_present: bool
    gnd_label_present: bool
    tran_fields: Tuple[Dict[str, str], ...]


def parse_spice_netlist(raw_lines: Sequence[str], netlist_path: str = "") -> ParsedNetlist:
    """Parse a SPICE netlist into conservative deterministic facts."""
    active_lines: List[str] = []
    comment_lines: List[str] = []
    ignored_comment_component_like_lines: List[str] = []
    components: List[ComponentFact] = []
    nodes = set()
    ordinary_directives: List[str] = []
    analysis_directives: List[str] = []
    control_block_lines: List[str] = []
    control_commands: List[str] = []
    output_commands: List[str] = []
    includes: List[IncludeFact] = []
    model_names: List[str] = []
    subckt_definitions: List[str] = []
    subckt_calls: List[SubcircuitCallFact] = []
    voltage_sources: List[str] = []
    load_candidates: List[str] = []
    tran_fields: List[Dict[str, str]] = []
    in_control_block = False
    netlist_dir = os.path.dirname(os.path.abspath(netlist_path)) if netlist_path else ""

    for raw in _logical_lines(raw_lines):
        stripped_raw = raw.strip()
        if not stripped_raw:
            continue

        if stripped_raw.startswith("*"):
            comment_lines.append(stripped_raw)
            if _is_component_like_line(stripped_raw):
                ignored_comment_component_like_lines.append(stripped_raw)
            continue

        line = _strip_spice_inline_comment(stripped_raw)
        if not line:
            continue

        tokens = line.split()
        first = tokens[0]
        lower_first = first.lower()
        active_lines.append(line)

        if in_control_block:
            control_block_lines.append(line)
            if lower_first != ".endc":
                control_commands.append(line)
                if lower_first in _CONTROL_OUTPUT_COMMANDS:
                    output_commands.append(line)
            if lower_first == ".endc":
                in_control_block = False
            continue

        if lower_first == ".control":
            control_block_lines.append(line)
            in_control_block = True
            continue

        if first.startswith("."):
            if lower_first in _ANALYSIS_DIRECTIVES:
                analysis_directives.append(line)
                if lower_first == ".tran":
                    tran_fields.append(_parse_tran_directive(line))
            else:
                ordinary_directives.append(line)

            if lower_first == ".include" and len(tokens) >= 2:
                path_part = line[len(first):].strip().strip('\'"')
                includes.append(_include_fact(path_part, netlist_dir))
            elif lower_first == ".model" and len(tokens) >= 2:
                model_names.append(tokens[1])
            elif lower_first == ".subckt" and len(tokens) >= 2:
                subckt_definitions.append(tokens[1])
            elif lower_first in _OUTPUT_DIRECTIVES:
                output_commands.append(line)
            continue

        if first[0].upper() in _COMPONENT_PREFIXES:
            component = _component_fact(tokens, line)
            components.append(component)
            nodes.update(component.nodes)

            if component.prefix == "V" and len(tokens) >= 4:
                voltage_sources.append(
                    f"{component.reference} between {tokens[1]} and {tokens[2]} uses {' '.join(tokens[3:])}"
                )
            if component.prefix == "R":
                load = _load_candidate_fact(tokens)
                if load:
                    load_candidates.append(load)
            if component.value_or_model and component.prefix in {"D", "Q", "J", "M"}:
                model_names.append(component.value_or_model)
            if component.prefix == "X" and component.value_or_model:
                subckt_calls.append(
                    SubcircuitCallFact(
                        reference=component.reference,
                        subcircuit=component.value_or_model,
                        nodes=component.nodes,
                        raw=component.raw,
                    )
                )

    defined_subckts = {name.lower() for name in subckt_definitions}
    unresolved_subckt_calls: List[SubcircuitCallFact] = []
    if not includes:
        unresolved_subckt_calls = [
            call for call in subckt_calls
            if call.subcircuit.lower() not in defined_subckts
        ]

    sorted_nodes = tuple(sorted(nodes, key=lambda item: item.lower()))
    return ParsedNetlist(
        filename=os.path.basename(netlist_path) if netlist_path else "",
        path=netlist_path,
        total_lines=len(raw_lines),
        active_lines=tuple(active_lines),
        comment_lines=tuple(comment_lines),
        ignored_comment_component_like_lines=tuple(ignored_comment_component_like_lines),
        components=tuple(components),
        nodes=sorted_nodes,
        ordinary_directives=tuple(ordinary_directives),
        analysis_directives=tuple(analysis_directives),
        control_block_lines=tuple(control_block_lines),
        control_commands=tuple(control_commands),
        output_commands=tuple(output_commands),
        includes=tuple(includes),
        model_names=tuple(sorted(set(model_names), key=lambda item: item.lower())),
        subckt_definitions=tuple(sorted(set(subckt_definitions), key=lambda item: item.lower())),
        subckt_calls=tuple(subckt_calls),
        unresolved_subckt_calls=tuple(unresolved_subckt_calls),
        voltage_sources=tuple(voltage_sources),
        load_candidates=tuple(load_candidates),
        reference_node_0_present=any(node == "0" for node in sorted_nodes),
        gnd_label_present=any(node.lower() == "gnd" for node in sorted_nodes),
        tran_fields=tuple(tran_fields),
    )


NETLIST_SYSTEM_PROMPT = (
    "You are an expert electronics engineer assistant inside eSim.\n"
    "Your job is to provide a helpful, conversational analysis of the circuit based strictly on the provided structured facts. Do not guess the circuit's overall function since you cannot see the exact wiring.\n\n"
    "Please structure your response into these sections:\n"
    "1. Overview: Provide a friendly summary of the components used in the circuit based on the component counts.\n"
    "2. Simulation Setup: Explain the simulation directives in plain English (e.g., 'You are running a transient analysis for 10ms').\n"
    "3. Analysis & Recommendations: Look at the OBVIOUS_ISSUES section from the facts. Explain WHY these issues are problematic in SPICE. For example, if the ground is missing, explain that SPICE needs a 0V reference point. If there are no issues, just tell the user the circuit looks structurally sound.\n\n"
    "CRITICAL RULES:\n"
    "- Only rely on the provided facts.\n"
    "- Do not invent or assume the circuit's purpose.\n"
    "- If the user asks about an invalid or unknown parameter in a follow-up (e.g., 'AA'), explicitly state that it is a syntax error. Do NOT invent a meaning for it."
)


def build_netlist_summary_prompt(
        parsed: ParsedNetlist, raw_lines: Sequence[str]) -> str:
    """Build a data-only grounding prompt for the LLM from deterministic facts.

    Instructions are in NETLIST_SYSTEM_PROMPT (sent as the system message).
    This prompt contains only structured facts. The raw netlist is intentionally
    excluded to prevent the LLM from hallucinating fixes for syntax errors.
    """
    fact_block = "\n".join(build_netlist_facts(parsed, raw_lines))

    return (
        "[YAML FACTS]\n"
        f"{fact_block}\n"
        "[END_ESIM_NETLIST_CONTEXT]\n\n"
        "CRITICAL INSTRUCTION FOR LLM:\n"
        "Act as a helpful AI assistant. Provide a conversational, friendly analysis of the facts above. Explain the components, explain the simulation setup, and most importantly, explain WHY any listed OBVIOUS_ISSUES are problematic in a SPICE simulation.\n"
        "Do not guess the circuit's overall function. If the user later asks about invalid syntax (like 'AA'), do not invent a meaning; explicitly state it is a syntax error."
    )


def build_netlist_facts(parsed: ParsedNetlist, raw_lines: Sequence[str]) -> List[str]:
    include_statuses = [
        f"{item.token}: {'FOUND' if item.exists else 'MISSING'} at {item.resolved_path}"
        for item in parsed.includes
    ]
    component_lines = [component.raw for component in parsed.components]
    component_type_counts = _component_type_counts(parsed.components)
    subckt_calls = [
        f"{call.reference} instantiates {call.subcircuit} with nodes ({', '.join(call.nodes)})"
        for call in parsed.subckt_calls
    ]
    unresolved_subckt_calls = [
        f"{call.reference} instantiates {call.subcircuit}"
        for call in parsed.unresolved_subckt_calls
    ]
    tran_fields = []
    for tran in parsed.tran_fields:
        for key in ("TRAN_RAW", "TRAN_TSTEP", "TRAN_TSTOP", "TRAN_TSTART", "TRAN_TMAX"):
            if key in tran:
                tran_fields.append(f"{key}: {tran[key]}")

    obvious_issues = []
    if not parsed.reference_node_0_present and not parsed.gnd_label_present:
        obvious_issues.append("Missing reference ground (node '0' or 'GND').")
    if not parsed.analysis_directives:
        obvious_issues.append("No simulation directives (e.g. .tran, .dc, .ac) found.")
    if parsed.unresolved_subckt_calls:
        obvious_issues.append(f"Unresolved subcircuits: {', '.join(c.subcircuit for c in parsed.unresolved_subckt_calls)}")
    missing_includes = [item.token for item in parsed.includes if not item.exists]
    if missing_includes:
        obvious_issues.append(f"Missing included files: {', '.join(missing_includes)}")
    
    if not obvious_issues:
        obvious_issues = ["None"]

    return [
        _fact_line("NETLIST_FILE", parsed.filename),
        _fact_line("NETLIST_PATH", parsed.path),
        _fact_line("TOTAL_LINES", len(raw_lines)),
        _fact_line("ACTIVE_LINE_COUNT", len(parsed.active_lines)),
        _fact_line("COMMENT_LINE_COUNT", len(parsed.comment_lines)),
        _fact_line("IGNORED_COMMENT_COMPONENT_LIKE_LINES", parsed.ignored_comment_component_like_lines),
        _fact_line("COMPONENT_COUNT", len(parsed.components)),
        _fact_line("COMPONENT_TYPE_COUNTS", component_type_counts),
        _fact_line("CURRENT_SOURCE_COUNT", component_type_counts.get("I", 0)),
        _fact_line("COMPONENT_LINES", component_lines),
        _fact_line("NODES", parsed.nodes),
        _fact_line("ORDINARY_DIRECTIVES", parsed.ordinary_directives),
        _fact_line("ANALYSIS_DIRECTIVES", parsed.analysis_directives),
        _fact_line("CONTROL_BLOCK_LINES", parsed.control_block_lines),
        _fact_line("CONTROL_COMMANDS", parsed.control_commands),
        _fact_line("OUTPUT_COMMANDS", parsed.output_commands),
        _fact_line("INCLUDE_FILES", [item.token for item in parsed.includes]),
        _fact_line("INCLUDE_FILE_STATUSES", include_statuses),
        _fact_line("MODEL_NAMES", parsed.model_names),
        _fact_line("SUBCKT_DEFINITIONS", parsed.subckt_definitions),
        _fact_line("SUBCKT_CALLS", subckt_calls),
        _fact_line("VOLTAGE_SOURCES", parsed.voltage_sources),
        _fact_line("LOAD_CANDIDATES", parsed.load_candidates),
        _fact_line("UNRESOLVED_SUBCKT_CALLS", unresolved_subckt_calls),
        _fact_line("SPICE_REFERENCE_NODE_0_PRESENT", parsed.reference_node_0_present),
        _fact_line("GND_LABEL_PRESENT", parsed.gnd_label_present),
        _fact_line("TRAN_FIELDS", tran_fields),
        _fact_line("OBVIOUS_ISSUES", obvious_issues),
    ]


def _component_type_counts(components: Sequence[ComponentFact]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for component in components:
        counts[component.prefix] = counts.get(component.prefix, 0) + 1
    return counts


def _logical_lines(raw_lines: Sequence[str]) -> Iterable[str]:
    logical: List[str] = []
    for raw in raw_lines:
        line = raw.rstrip("\r\n")
        stripped = line.lstrip()
        if stripped.startswith("+") and logical:
            logical[-1] = f"{logical[-1]} {stripped[1:].strip()}"
        else:
            logical.append(line)
    return logical


def _strip_spice_inline_comment(line: str) -> str:
    for index, char in enumerate(line):
        if char == ";":
            return line[:index].strip()
        if char == "$" and (index == 0 or line[index - 1].isspace()):
            return line[:index].strip()
    return line.strip()


def _component_fact(tokens: Sequence[str], raw: str) -> ComponentFact:
    reference = tokens[0]
    prefix = reference[0].upper()
    nodes, value_or_model = _component_nodes_and_value(tokens)
    return ComponentFact(
        reference=reference,
        prefix=prefix,
        nodes=tuple(nodes),
        value_or_model=value_or_model,
        raw=raw,
    )


def _component_nodes_and_value(tokens: Sequence[str]) -> Tuple[List[str], str]:
    if len(tokens) < 2:
        return [], ""

    prefix = tokens[0][0].upper()
    if prefix == "X" and len(tokens) >= 3:
        return list(tokens[1:-1]), tokens[-1]
    if prefix in {"R", "C", "L", "V", "I"} and len(tokens) >= 4:
        return list(tokens[1:3]), " ".join(tokens[3:])
    if prefix == "D" and len(tokens) >= 4:
        return list(tokens[1:3]), tokens[3]
    if prefix in {"Q", "J"} and len(tokens) >= 5:
        return list(tokens[1:4]), tokens[4]
    if prefix == "M" and len(tokens) >= 6:
        return list(tokens[1:5]), tokens[5]
    if prefix in {"E", "G"} and len(tokens) >= 5:
        return list(tokens[1:5]), " ".join(tokens[5:])
    if prefix in {"F", "H"} and len(tokens) >= 4:
        return list(tokens[1:3]), " ".join(tokens[3:])

    return list(tokens[1:-1]) if len(tokens) > 2 else list(tokens[1:]), (
        tokens[-1] if len(tokens) > 2 else ""
    )


def _normalize_include_path(token: str) -> str:
    return token.strip().strip('"').strip("'")


def _include_fact(include_token: str, netlist_dir: str) -> IncludeFact:
    include_path = _normalize_include_path(include_token)
    if netlist_dir and include_path and not os.path.isabs(include_path):
        resolved_path = os.path.abspath(os.path.join(netlist_dir, include_path))
    else:
        resolved_path = include_path
    return IncludeFact(
        token=include_path,
        resolved_path=resolved_path,
        exists=bool(resolved_path and os.path.exists(resolved_path)),
    )


def _is_component_like_line(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("*"):
        stripped = stripped[1:].lstrip()
    if not stripped:
        return False
    tokens = stripped.split()
    first = tokens[0]
    if first.startswith("."):
        return True

    min_tokens = {
        "R": 4, "C": 4, "L": 4, "V": 4, "I": 4, "D": 4,
        "Q": 5, "J": 5, "M": 6, "E": 5, "G": 5, "F": 4,
        "H": 4, "K": 4, "T": 4, "U": 3, "W": 4, "X": 4,
        "Z": 4,
    }
    return len(tokens) >= min_tokens.get(first[0].upper(), 99)


def _parse_tran_directive(line: str) -> Dict[str, str]:
    tokens = line.split()
    parsed = {"TRAN_RAW": line}
    fields = [
        ("TRAN_TSTEP", 1),
        ("TRAN_TSTOP", 2),
        ("TRAN_TSTART", 3),
        ("TRAN_TMAX", 4),
    ]
    for name, index in fields:
        if len(tokens) > index:
            parsed[name] = _format_time_fact(tokens[index])
    return parsed


def _format_time_fact(value: str) -> str:
    seconds = _spice_number_to_float(value)
    if seconds is None:
        return value
    return f"{value} = {seconds:g} s ({_format_seconds(seconds)})"


def _spice_number_to_float(value: str):
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass

    match = re.match(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)([a-z]+)$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None

    base = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix.startswith("meg"):
        scale_char = "meg"
    else:
        scale_char = suffix[0]

    scale = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }.get(scale_char)
    return None if scale is None else base * scale


def _format_seconds(seconds: float) -> str:
    if seconds == 0:
        return "0 s"

    abs_seconds = abs(seconds)
    units = [
        (1.0, "s"),
        (1e-3, "ms"),
        (1e-6, "us"),
        (1e-9, "ns"),
        (1e-12, "ps"),
    ]
    for scale, unit in units:
        value = seconds / scale
        if abs_seconds >= scale and abs(value) < 1000:
            return f"{value:g} {unit}"
    return f"{seconds:g} s"


def _load_candidate_fact(tokens: Sequence[str]) -> str:
    if len(tokens) < 4:
        return ""
    node_a, node_b = tokens[1], tokens[2]
    if node_a.lower() not in ("0", "gnd") and node_b.lower() not in ("0", "gnd"):
        return ""
    return f"{tokens[0]} {node_a}-{node_b} {tokens[3]}"


def _fact_line(name: str, values) -> str:
    if isinstance(values, bool):
        rendered = "Yes" if values else "No"
    elif isinstance(values, (list, tuple, set)):
        values = list(values)
        rendered = "None" if not values else ", ".join(str(v) for v in values[:MAX_NETLIST_FACT_ITEMS])
        if len(values) > MAX_NETLIST_FACT_ITEMS:
            rendered += f", ... ({len(values) - MAX_NETLIST_FACT_ITEMS} more)"
    else:
        rendered = str(values)
    
    # Format name nicely (e.g. replace underscores with spaces and title case)
    formatted_name = name.replace("_", " ").title()
    return f"{formatted_name}: {rendered}"


def _bounded_text(lines: Sequence[str]) -> Tuple[str, bool]:
    text = "\n".join(lines)
    if len(text) <= MAX_NETLIST_CONTEXT_CHARS:
        return text, False
    return text[:MAX_NETLIST_CONTEXT_CHARS], True
