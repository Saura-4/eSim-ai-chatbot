"""Regression tests for the NgSpice error-analysis pipeline.

Covers:
  1. ERR001 Missing Model
  2. ERR002 Missing Subcircuit
  3. ERR006 Invalid Model Syntax
  4. ERR011 Source Loop
  5. ERR012 Invalid Parameter
  6. Multiple root-cause detection (ERR001 + ERR002)
  7. ERR006 suppressing ERR001
  8. ERR002 suppressing generic workspace/circuit-loading failures

Run:  python test_error_pipeline_regression.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.error_patterns import match_error_patterns, ErrorMatch
from chatbot.error_log_analysis import rank_errors, RankedError

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


def _find_match(matches, error_id):
    """Find a match by error_id."""
    for m in matches:
        if m.error_id == error_id:
            return m
    return None


def _find_ranked(ranked, error_id):
    """Find a ranked error by error_id."""
    for r in ranked:
        if r.match.error_id == error_id:
            return r
    return None


# ── Test 1: ERR001 Missing Model ─────────────────────────────────────────────

def test_err001_missing_model():
    print("\n=== Test 1: ERR001 Missing Model ===")
    log = (
        "warning, can't find model 'mydiode'\n"
        "could not find a valid modelname\n"
    )
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m = _find_match(matches, "ERR001")
    _assert(m is not None, "ERR001 detected")
    _assert(m.category == "Missing Model", "Category is Missing Model")
    _assert(m.extracted_facts.get("Model") == "mydiode", "Model name extracted")
    _assert(
        m.extracted_facts.get("Component") == "Unknown",
        "Component is Unknown (not hallucinated)",
    )

    r = _find_ranked(ranked, "ERR001")
    _assert(r is not None and r.role == "Root Cause", "ERR001 is Root Cause")


# ── Test 2: ERR002 Missing Subcircuit ─────────────────────────────────────────

def test_err002_missing_subcircuit():
    print("\n=== Test 2: ERR002 Missing Subcircuit ===")
    log = "Error: unknown subckt: x1 net-*c1-pad1* 0 out lm7805\n"
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m = _find_match(matches, "ERR002")
    _assert(m is not None, "ERR002 detected")
    _assert(m.category == "Missing Subcircuit", "Category is Missing Subcircuit")
    _assert(m.extracted_facts.get("Subcircuit") == "lm7805", "Subcircuit name extracted")
    _assert(m.extracted_facts.get("Component") == "x1", "Component name extracted")

    r = _find_ranked(ranked, "ERR002")
    _assert(r is not None and r.role == "Root Cause", "ERR002 is Root Cause")


# ── Test 3: ERR006 Invalid Model Syntax ──────────────────────────────────────

def test_err006_invalid_model_syntax():
    print("\n=== Test 3: ERR006 Invalid Model Syntax ===")
    log = "Unknown model type xyz - ignored\n"
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m = _find_match(matches, "ERR006")
    _assert(m is not None, "ERR006 detected")
    _assert(m.category == "Invalid Model Syntax", "Category is Invalid Model Syntax")
    _assert(m.extracted_facts.get("Invalid Type") == "xyz", "Invalid type extracted")

    r = _find_ranked(ranked, "ERR006")
    _assert(r is not None and r.role == "Root Cause", "ERR006 is Root Cause")


# ── Test 4: ERR011 Source Loop ────────────────────────────────────────────────

def test_err011_source_loop():
    print("\n=== Test 4: ERR011 Source Loop ===")
    log = (
        "v1 out nloop dc 5\n"
        "v2 nloop out dc 5\n"
        "voltage source loop detected\n"
        "Error: singular matrix\n"
        "timestep too small\n"
    )
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m = _find_match(matches, "ERR011")
    _assert(m is not None, "ERR011 detected")
    _assert(m.category == "Source Loop", "Category is Source Loop")
    # Should have extracted source names
    sources = m.extracted_facts.get("Sources", "")
    _assert("v1" in sources or "V1" in sources.upper(), "v1 found in sources")
    _assert("v2" in sources or "V2" in sources.upper(), "v2 found in sources")

    # Ranking: ERR011 = Root Cause, ERR004 + ERR005 suppressed
    r011 = _find_ranked(ranked, "ERR011")
    _assert(r011 is not None and r011.role == "Root Cause", "ERR011 is Root Cause")

    r004 = _find_ranked(ranked, "ERR004")
    _assert(
        r004 is not None and r004.role != "Root Cause",
        "ERR004 is NOT Root Cause (suppressed by ERR011)",
    )
    _assert(
        r004 is not None and r004.suppressed_by == "ERR011",
        "ERR004 suppressed_by = ERR011",
    )

    r005 = _find_ranked(ranked, "ERR005")
    _assert(
        r005 is not None and r005.role != "Root Cause",
        "ERR005 is NOT Root Cause (suppressed by ERR011)",
    )
    _assert(
        r005 is not None and r005.suppressed_by == "ERR011",
        "ERR005 suppressed_by = ERR011",
    )


# ── Test 5: ERR012 Invalid Parameter ─────────────────────────────────────────

def test_err012_invalid_parameter():
    print("\n=== Test 5: ERR012 Invalid Parameter ===")
    log = "unknown parameter foo\n"
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m = _find_match(matches, "ERR012")
    _assert(m is not None, "ERR012 detected")
    _assert(
        m.category == "Invalid Parameter / Syntax Error",
        "Category is Invalid Parameter / Syntax Error",
    )
    _assert(
        m.extracted_facts.get("Reason") == "Unknown parameter",
        "Reason is 'Unknown parameter' (not numeric)",
    )
    _assert(
        m.extracted_facts.get("Parameter") == "foo",
        "Parameter name 'foo' extracted",
    )

    r = _find_ranked(ranked, "ERR012")
    _assert(r is not None and r.role == "Root Cause", "ERR012 is Root Cause")


# ── Test 6: Multiple Root Causes (ERR001 + ERR002) ───────────────────────────

def test_multiple_root_causes():
    print("\n=== Test 6: Multiple Root Causes (ERR001 + ERR002) ===")
    log = (
        "warning, can't find model 'mydiode'\n"
        "Error: unknown subckt: x1 net-*c1-pad1* 0 out lm7805\n"
    )
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    m001 = _find_match(matches, "ERR001")
    m002 = _find_match(matches, "ERR002")
    _assert(m001 is not None, "ERR001 detected")
    _assert(m002 is not None, "ERR002 detected")

    # Both should be Root Cause (both priority 1, no suppression between them)
    r001 = _find_ranked(ranked, "ERR001")
    r002 = _find_ranked(ranked, "ERR002")
    _assert(r001 is not None and r001.role == "Root Cause", "ERR001 is Root Cause")
    _assert(r002 is not None and r002.role == "Root Cause", "ERR002 is Root Cause")

    root_count = sum(1 for r in ranked if r.role == "Root Cause")
    _assert(root_count >= 2, f"At least 2 root causes reported (got {root_count})")


# ── Test 7: ERR006 suppresses ERR001 ─────────────────────────────────────────

def test_err006_suppresses_err001():
    print("\n=== Test 7: ERR006 suppresses ERR001 ===")
    log = (
        "Unknown model type xyz - ignored\n"
        "could not find a valid modelname\n"
    )
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    r006 = _find_ranked(ranked, "ERR006")
    r001 = _find_ranked(ranked, "ERR001")

    _assert(r006 is not None and r006.role == "Root Cause", "ERR006 is Root Cause")
    _assert(
        r001 is not None and r001.role != "Root Cause",
        "ERR001 is NOT Root Cause (suppressed)",
    )
    _assert(
        r001 is not None and r001.suppressed_by == "ERR006",
        "ERR001 suppressed_by = ERR006",
    )


# ── Test 8: ERR002 suppresses generic workspace failures ─────────────────────

def test_err002_suppresses_generic():
    print("\n=== Test 8: ERR002 suppresses generic workspace failures ===")
    log = (
        "Error: unknown subckt: x1 net-*c1-pad1* 0 out lm7805\n"
        "Error: there aren't any circuits loaded.\n"
        "Simulation Failed!\n"
    )
    matches = match_error_patterns(log)
    ranked = rank_errors(matches)

    r002 = _find_ranked(ranked, "ERR002")
    _assert(r002 is not None and r002.role == "Root Cause", "ERR002 is Root Cause")
    _assert(
        r002.match.extracted_facts.get("Component") == "x1",
        "Component x1 extracted",
    )
    _assert(
        r002.match.extracted_facts.get("Subcircuit") == "lm7805",
        "Subcircuit lm7805 extracted",
    )

    # Ensure no generic loading failure became a root cause
    for r in ranked:
        if r.role == "Root Cause" and r.match.error_id != "ERR002":
            _assert(
                False,
                f"Unexpected root cause: {r.match.category} ({r.match.error_id})",
            )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_err001_missing_model()
    test_err002_missing_subcircuit()
    test_err006_invalid_model_syntax()
    test_err011_source_loop()
    test_err012_invalid_parameter()
    test_multiple_root_causes()
    test_err006_suppresses_err001()
    test_err002_suppresses_generic()

    print(f"\n{'=' * 50}")
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
        sys.exit(0)
