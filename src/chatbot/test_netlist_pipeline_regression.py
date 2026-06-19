"""Regression tests for the netlist analysis pipeline.

Covers:
  1. Component parsing and tracking.
  2. Subcircuit parsing.
  3. Circuit block detection (Bridge Rectifier, Voltage Regulator, Filter Cap).
  4. Prompt building formatting logic.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.netlist_analysis import (
    parse_spice_netlist,
    detect_circuit_blocks,
    build_netlist_summary_prompt,
    format_netlist_table
)

PASS = 0
FAIL = 0


def _assert(condition: bool, msg: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {msg}")
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def test_basic_parsing():
    print("\n=== Test 1: Basic Component Parsing ===")
    netlist = [
        "* A basic netlist",
        "R1 1 0 10k",
        "C1 1 0 1uF",
        "V1 1 0 dc 5",
        ".tran 1m 10m"
    ]
    parsed = parse_spice_netlist(netlist)
    _assert(len(parsed.components) == 3, "Found 3 components")
    _assert(len(parsed.nodes) == 2, "Found 2 nodes (0, 1)")
    _assert(parsed.reference_node_0_present, "Node 0 is detected")
    _assert(len(parsed.analysis_directives) == 1, "Found 1 analysis directive")
    
    # Check components
    r1 = next(c for c in parsed.components if c.reference == "R1")
    _assert(r1.value_or_model == "10k", "R1 value is 10k")


def test_bridge_rectifier_detection():
    print("\n=== Test 2: Bridge Rectifier Detection ===")
    netlist = [
        "D1 AC1 DC+ 1N4007",
        "D2 AC2 DC+ 1N4007",
        "D3 DC- AC1 1N4007",
        "D4 DC- AC2 1N4007",
        "Vac1 AC1 0 sine(0 12 50)",
        "Vac2 AC2 0 sine(0 -12 50)",
        "Rload DC+ DC- 1k"
    ]
    parsed = parse_spice_netlist(netlist)
    blocks = detect_circuit_blocks(parsed)
    
    bridge_found = False
    for block in blocks:
        name, conf, details = block
        if "Bridge rectifier stage" in name:
            bridge_found = True
            _assert(conf == "HIGH", "Confidence is HIGH")
            _assert("DC+" in details and "DC-" in details, "Correct nodes identified")
    _assert(bridge_found, "Bridge rectifier detected")


def test_regulator_detection():
    print("\n=== Test 3: 7805 Regulator Detection ===")
    netlist = [
        "X1 IN 0 OUT LM7805",
        "R1 OUT 0 1k"
    ]
    parsed = parse_spice_netlist(netlist)
    blocks = detect_circuit_blocks(parsed)
    
    reg_found = False
    for block in blocks:
        name, conf, details = block
        if "7805" in name:
            reg_found = True
            _assert(conf == "HIGH", "Confidence is HIGH")
            _assert("IN, 0, OUT" in details, "Correct nodes identified")
    _assert(reg_found, "7805 regulator detected")


def test_filter_cap_detection():
    print("\n=== Test 4: Filter Capacitor Detection ===")
    netlist = [
        "C1 OUT 0 10u",
        "R1 OUT 0 1k"
    ]
    parsed = parse_spice_netlist(netlist)
    blocks = detect_circuit_blocks(parsed)
    
    cap_found = False
    for block in blocks:
        name, conf, details = block
        if "Filter capacitor" in name:
            cap_found = True
            _assert(conf == "HIGH", "Confidence is HIGH")
            _assert("OUT" in details and "0" in details, "Nodes correctly identified")
    _assert(cap_found, "Filter capacitor detected")


def test_markdown_table_formatting():
    print("\n=== Test 5: Markdown Table Formatting ===")
    netlist = [
        "R1 1 0 10k",
        "C1 1 0 1uF",
        ".tran 1m 10m"
    ]
    parsed = parse_spice_netlist(netlist)
    table = format_netlist_table(parsed)
    _assert("Resistors" in table, "Table includes Resistors")
    _assert("Capacitors" in table, "Table includes Capacitors")
    _assert("Transient Analysis" in table, "Table includes Transient Analysis")


if __name__ == "__main__":
    test_basic_parsing()
    test_bridge_rectifier_detection()
    test_regulator_detection()
    test_filter_cap_detection()
    test_markdown_table_formatting()

    print(f"\n{'=' * 50}")
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
        sys.exit(0)
