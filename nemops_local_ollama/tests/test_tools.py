"""Tests for NemOps tool functions."""

import json

import pytest

from nemops.tools.gpu_health import gpu_health_check
from nemops.tools.diagnostics import run_diagnostic
from nemops.tools.alert_gen import generate_alert


class TestGPUHealthCheck:
    """Test gpu_health_check tool."""

    def test_returns_gpus_and_summary(self):
        result = gpu_health_check()
        assert "gpus" in result
        assert "summary" in result
        assert len(result["gpus"]) > 0

    def test_gpu_has_required_fields(self):
        result = gpu_health_check()
        gpu = result["gpus"][0]
        required = [
            "gpu_id", "name", "temperature_c", "utilization_pct",
            "memory_used_gb", "memory_total_gb", "power_draw_w",
            "power_limit_w", "ecc_errors", "status", "issues",
        ]
        for field in required:
            assert field in gpu, f"Missing field: {field}"

    def test_single_gpu_check(self):
        result = gpu_health_check(gpu_id=0)
        assert "gpus" in result
        # Should have at least the requested GPU
        assert any(g["gpu_id"] == 0 for g in result["gpus"])

    def test_summary_counts_are_valid(self):
        result = gpu_health_check()
        s = result["summary"]
        total = s["healthy"] + s["warnings"] + s["critical"]
        assert total == s["total_gpus"]
        assert s["overall_status"] in ("healthy", "warning", "critical")


class TestDiagnostics:
    """Test run_diagnostic tool."""

    @pytest.mark.parametrize("test_name", [
        "memory_stress", "compute_stress", "nvlink_check",
        "pcie_bandwidth", "thermal_profile",
    ])
    def test_all_diagnostics_return_results(self, test_name):
        result = run_diagnostic(test_name, gpu_id=0)
        assert "test" in result
        assert "gpu_id" in result
        assert "passed" in result
        assert "duration_seconds" in result
        assert "details" in result
        assert "conclusion" in result
        assert result["test"] == test_name

    def test_unknown_diagnostic(self):
        result = run_diagnostic("nonexistent_test")
        assert "error" in result
        assert "available_tests" in result


class TestAlertGenerator:
    """Test generate_alert tool."""

    def test_critical_alert(self):
        result = generate_alert(
            severity="critical",
            title="GPU ECC Failure",
            analysis="HBM3 degradation detected",
            remediation_steps=["Drain GPU", "Replace hardware"],
        )
        alert = result["alert"]
        assert alert["severity"] == "critical"
        assert alert["title"] == "GPU ECC Failure"
        assert result["dispatched"] is True
        assert alert["escalation"]["policy"] == "gpu-critical-oncall"

    def test_warning_alert(self):
        result = generate_alert(
            severity="warning",
            title="Thermal Warning",
            analysis="Temperature elevated",
        )
        assert result["alert"]["severity"] == "warning"
        assert result["alert"]["escalation"]["response_sla_minutes"] == 60

    def test_invalid_severity_defaults_to_info(self):
        result = generate_alert(
            severity="INVALID",
            title="Test",
            analysis="Test",
        )
        assert result["alert"]["severity"] == "info"

    def test_alert_has_id_and_timestamp(self):
        result = generate_alert(
            severity="info",
            title="Test Alert",
            analysis="Testing",
        )
        alert = result["alert"]
        assert alert["alert_id"].startswith("NEMOPS-")
        assert "timestamp" in alert
        assert alert["source"] == "nemops-agent"
