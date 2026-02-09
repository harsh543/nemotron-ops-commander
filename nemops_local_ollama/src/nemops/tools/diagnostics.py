"""GPU diagnostic tools — simulated tests for demo mode."""

import random
import time
from typing import Any


def run_diagnostic(test_name: str, gpu_id: int = 0) -> dict[str, Any]:
    """Run a targeted GPU diagnostic test.

    Args:
        test_name: One of memory_stress, compute_stress, nvlink_check,
                   pcie_bandwidth, thermal_profile.
        gpu_id: GPU index to test.

    Returns:
        Dict with test results, pass/fail status, and details.
    """
    tests = {
        "memory_stress": _memory_stress,
        "compute_stress": _compute_stress,
        "nvlink_check": _nvlink_check,
        "pcie_bandwidth": _pcie_bandwidth,
        "thermal_profile": _thermal_profile,
    }

    if test_name not in tests:
        return {
            "error": f"Unknown test: {test_name}",
            "available_tests": list(tests.keys()),
        }

    return tests[test_name](gpu_id)


def _memory_stress(gpu_id: int) -> dict[str, Any]:
    """Simulate memory stress test — allocate and verify patterns."""
    ecc_errors = random.choice([0, 0, 0, 0, 2, 15])
    passed = ecc_errors == 0

    return {
        "test": "memory_stress",
        "gpu_id": gpu_id,
        "passed": passed,
        "duration_seconds": round(random.uniform(30, 120), 1),
        "details": {
            "memory_tested_gb": 80.0,
            "pattern": "walking-ones",
            "passes": 3,
            "ecc_errors_during_test": ecc_errors,
            "pages_retired": ecc_errors // 2,
        },
        "conclusion": (
            "Memory stress test PASSED — no ECC errors detected"
            if passed
            else f"Memory stress test FAILED — {ecc_errors} ECC errors detected. "
            f"HBM3 degradation confirmed. {ecc_errors // 2} pages retired."
        ),
    }


def _compute_stress(gpu_id: int) -> dict[str, Any]:
    """Simulate compute stress test — matrix operations."""
    peak_temp = random.uniform(70, 95)
    throttled = peak_temp > 85
    passed = not throttled

    return {
        "test": "compute_stress",
        "gpu_id": gpu_id,
        "passed": passed,
        "duration_seconds": round(random.uniform(60, 180), 1),
        "details": {
            "operation": "FP16 GEMM (4096x4096)",
            "sustained_tflops": round(
                random.uniform(300, 990) if not throttled else random.uniform(150, 300), 1
            ),
            "peak_temperature_c": round(peak_temp, 1),
            "thermal_throttle_events": random.randint(5, 50) if throttled else 0,
            "clock_boost_sustained": not throttled,
        },
        "conclusion": (
            "Compute stress PASSED — sustained peak performance, no throttling"
            if passed
            else f"Compute stress WARNING — thermal throttling at {peak_temp:.0f}°C, "
            f"performance reduced. Check cooling system."
        ),
    }


def _nvlink_check(gpu_id: int) -> dict[str, Any]:
    """Simulate NVLink connectivity check."""
    link_errors = random.choice([0, 0, 0, 12, 150])
    passed = link_errors == 0

    return {
        "test": "nvlink_check",
        "gpu_id": gpu_id,
        "passed": passed,
        "duration_seconds": round(random.uniform(5, 15), 1),
        "details": {
            "nvlink_version": 4,
            "links_active": 18 if passed else random.randint(14, 17),
            "links_total": 18,
            "bandwidth_gb_s": round(900.0 if passed else random.uniform(500, 750), 1),
            "crc_errors": link_errors,
            "replay_errors": link_errors // 3,
        },
        "conclusion": (
            "NVLink check PASSED — all 18 links active, 900 GB/s bandwidth"
            if passed
            else f"NVLink check FAILED — {link_errors} CRC errors, "
            f"only {18 - (link_errors // 30)} links healthy. Check NVSwitch."
        ),
    }


def _pcie_bandwidth(gpu_id: int) -> dict[str, Any]:
    """Simulate PCIe bandwidth test."""
    actual_bw = random.choice([63.0, 63.0, 63.0, 31.5, 15.75])
    expected_bw = 63.0  # PCIe Gen5 x16
    passed = actual_bw >= expected_bw * 0.9

    return {
        "test": "pcie_bandwidth",
        "gpu_id": gpu_id,
        "passed": passed,
        "duration_seconds": round(random.uniform(10, 30), 1),
        "details": {
            "pcie_gen": 5 if passed else (4 if actual_bw > 20 else 3),
            "pcie_width": 16 if passed else (8 if actual_bw > 20 else 4),
            "bandwidth_measured_gb_s": actual_bw,
            "bandwidth_expected_gb_s": expected_bw,
            "bandwidth_ratio": round(actual_bw / expected_bw, 2),
        },
        "conclusion": (
            f"PCIe bandwidth PASSED — {actual_bw} GB/s (Gen5 x16)"
            if passed
            else f"PCIe bandwidth DEGRADED — {actual_bw} GB/s vs expected {expected_bw} GB/s. "
            f"Link negotiated at reduced width/speed. Reseat GPU or check motherboard slot."
        ),
    }


def _thermal_profile(gpu_id: int) -> dict[str, Any]:
    """Simulate thermal profile analysis."""
    idle_temp = random.uniform(28, 45)
    load_temp = random.uniform(60, 95)
    fan_response = load_temp < 85

    return {
        "test": "thermal_profile",
        "gpu_id": gpu_id,
        "passed": fan_response and load_temp < 85,
        "duration_seconds": round(random.uniform(120, 300), 1),
        "details": {
            "idle_temperature_c": round(idle_temp, 1),
            "load_temperature_c": round(load_temp, 1),
            "thermal_headroom_c": round(92 - load_temp, 1),
            "fan_response_adequate": fan_response,
            "ambient_temperature_c": round(random.uniform(20, 30), 1),
            "thermal_paste_condition": "good" if load_temp < 80 else "degraded",
        },
        "conclusion": (
            f"Thermal profile GOOD — load temp {load_temp:.0f}°C with "
            f"{92 - load_temp:.0f}°C headroom"
            if load_temp < 85
            else f"Thermal profile CRITICAL — load temp {load_temp:.0f}°C approaching "
            f"shutdown threshold (92°C). Thermal paste may need replacement."
        ),
    }
