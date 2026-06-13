#!/usr/bin/env python3
"""Focused tests for deterministic netlist analysis helpers."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chatbot.netlist_analysis import (  # noqa:E402
    build_netlist_summary_prompt,
    parse_spice_netlist,
)


def _read_lines(path):
    with open(path, "r", errors="replace") as handle:
        return handle.readlines()


def test_7805_fixture():
    netlist_path = os.path.join(
        ROOT,
        "Examples",
        "7805VoltageRegulator",
        "7805VoltageRegulator.cir.out",
    )
    raw_lines = _read_lines(netlist_path)
    parsed = parse_spice_netlist(raw_lines, netlist_path)

    include_tokens = {item.token for item in parsed.includes}
    assert {"lm7805.sub", "D.lib"}.issubset(include_tokens)
    assert all(item.exists for item in parsed.includes)

    component_refs = {component.reference.lower() for component in parsed.components}
    assert {"d1", "d2", "d3", "d4", "d5", "c1", "c2", "r1", "v1", "x1"}.issubset(component_refs)
    assert "u1" not in component_refs
    assert any("u1  in1 plot_v1" in line.lower() for line in parsed.ignored_comment_component_like_lines)

    assert parsed.gnd_label_present is True
    assert parsed.reference_node_0_present is False
    assert any(call.reference.lower() == "x1" and call.subcircuit.lower() == "lm7805"
               for call in parsed.subckt_calls)
    assert not parsed.unresolved_subckt_calls

    assert any(source.startswith("v1 between in1 and in2 uses sine") for source in parsed.voltage_sources)
    assert any(load.lower().startswith("r1 out-gnd 1k") for load in parsed.load_candidates)
    assert ".tran 10e-03 100e-03 0e-00" in parsed.analysis_directives
    assert any(command == "run" for command in parsed.control_commands)
    assert any(command == "plot v(out)" for command in parsed.output_commands)

    prompt = build_netlist_summary_prompt(parsed, raw_lines)
    assert "[FACT COMPONENT_LINES=" in prompt
    assert "[FACT CURRENT_SOURCE_COUNT=0]" in prompt
    assert "Use the deterministic facts and bounded netlist text below as the source of truth." in prompt
    assert "Do not invent components, current sources, voltages, simulation results" in prompt
    assert "Keep the response concise and concrete." in prompt
    assert "Observed topology only" in prompt
    assert "do not call its definition missing" in prompt
    assert "describe what circuit this is and what it does" not in prompt
    assert "Please: (1)" not in prompt
    assert "Likely circuit intent" not in prompt
    assert "Unknown / cannot determine from netlist alone" in prompt


def test_unresolved_subcircuit_without_include():
    raw_lines = [
        "V1 in 0 DC 12\n",
        "X1 in 0 out missing_regulator\n",
        ".tran 1m 10m\n",
        ".end\n",
    ]
    parsed = parse_spice_netlist(raw_lines, "inline.cir.out")

    assert parsed.reference_node_0_present is True
    assert not parsed.includes
    assert len(parsed.subckt_calls) == 1
    assert len(parsed.unresolved_subckt_calls) == 1
    assert parsed.unresolved_subckt_calls[0].subcircuit == "missing_regulator"


def main():
    tests = [
        test_7805_fixture,
        test_unresolved_subcircuit_without_include,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
