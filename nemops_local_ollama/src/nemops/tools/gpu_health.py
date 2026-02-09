"""GPU health check tool — mock mode for demo, real mode for NVIDIA GPUs."""

import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class GPUHealthReport:
    """Structured GPU health metrics."""

    gpu_id: int
    name: str
    temperature_c: float
    utilization_pct: float
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization_pct: float
    power_draw_w: float
    power_limit_w: float
    clock_speed_mhz: int
    max_clock_mhz: int
    ecc_single_bit_errors: int
    ecc_double_bit_errors: int
    pcie_gen: int
    pcie_width: int
    xid_errors: list[int]
    fan_speed_pct: float
    status: str  # "healthy", "warning", "critical"
    issues: list[str]
    timestamp: float


GPU_MODE = os.getenv("GPU_MODE", "mock")

# Weighted mock scenarios based on real production data
# From experience: 10M+ events/day across 10,000+ GPUs
MOCK_SCENARIOS = {
    "healthy": {
        "weight": 0.55,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA H100 80GB HBM3 [GPU {gpu_id}]",
            temperature_c=random.uniform(35, 55),
            utilization_pct=random.uniform(60, 95),
            memory_used_gb=round(random.uniform(30, 65), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(37, 81), 1),
            power_draw_w=round(random.uniform(200, 350), 1),
            power_limit_w=700.0,
            clock_speed_mhz=random.randint(1800, 2100),
            max_clock_mhz=2100,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=5,
            pcie_width=16,
            xid_errors=[],
            fan_speed_pct=round(random.uniform(30, 50), 1),
            status="healthy",
            issues=[],
            timestamp=time.time(),
        ),
    },
    "thermal_warning": {
        "weight": 0.12,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA A100 80GB SXM4 [GPU {gpu_id}]",
            temperature_c=round(random.uniform(83, 92), 1),
            utilization_pct=round(random.uniform(30, 50), 1),  # Throttled
            memory_used_gb=round(random.uniform(40, 70), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(50, 87), 1),
            power_draw_w=round(random.uniform(250, 300), 1),
            power_limit_w=400.0,
            clock_speed_mhz=random.randint(900, 1200),  # Throttled
            max_clock_mhz=1410,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=4,
            pcie_width=16,
            xid_errors=[],
            fan_speed_pct=100.0,  # Fans maxed
            status="warning",
            issues=[
                "Temperature above 83°C threshold — GPU is thermal throttling",
                "Clock speed reduced to ~60% of max due to thermal limits",
                "Fan speed at 100% — check ambient temperature and airflow",
            ],
            timestamp=time.time(),
        ),
    },
    "ecc_errors": {
        "weight": 0.10,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA H100 80GB HBM3 [GPU {gpu_id}]",
            temperature_c=round(random.uniform(45, 60), 1),
            utilization_pct=round(random.uniform(70, 90), 1),
            memory_used_gb=round(random.uniform(50, 75), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(62, 93), 1),
            power_draw_w=round(random.uniform(300, 500), 1),
            power_limit_w=700.0,
            clock_speed_mhz=random.randint(1800, 2100),
            max_clock_mhz=2100,
            ecc_single_bit_errors=random.randint(50, 500),
            ecc_double_bit_errors=random.randint(1, 5),
            pcie_gen=5,
            pcie_width=16,
            xid_errors=[63, 63, 64],  # XID 63 = ECC row remap, XID 64 = ECC page retire
            fan_speed_pct=round(random.uniform(40, 60), 1),
            status="critical",
            issues=[
                "ECC double-bit errors detected — HBM3 degradation likely",
                "XID 63 (ECC row remapping) observed — rows being retired",
                "XID 64 (ECC page retirement) — memory pages permanently retired",
                "PREDICTIVE FAILURE: Based on error rate trajectory, GPU failure "
                "estimated in 10-15 days. Schedule replacement.",
            ],
            timestamp=time.time(),
        ),
    },
    "memory_pressure": {
        "weight": 0.08,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA A100 80GB SXM4 [GPU {gpu_id}]",
            temperature_c=round(random.uniform(50, 65), 1),
            utilization_pct=round(random.uniform(95, 100), 1),
            memory_used_gb=round(random.uniform(76, 79.5), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(95, 99), 1),
            power_draw_w=round(random.uniform(350, 400), 1),
            power_limit_w=400.0,
            clock_speed_mhz=random.randint(1200, 1410),
            max_clock_mhz=1410,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=4,
            pcie_width=16,
            xid_errors=[31],  # XID 31 = GPU memory page fault
            fan_speed_pct=round(random.uniform(60, 80), 1),
            status="warning",
            issues=[
                "GPU memory utilization >95% — OOM kill risk",
                "XID 31 (GPU memory page fault) — fragmentation detected",
                "Consider: torch.cuda.empty_cache() or reduce batch size",
            ],
            timestamp=time.time(),
        ),
    },
    "gpu_fallen_off_bus": {
        "weight": 0.05,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA H100 80GB HBM3 [GPU {gpu_id}]",
            temperature_c=0.0,  # Can't read
            utilization_pct=0.0,
            memory_used_gb=0.0,
            memory_total_gb=0.0,
            memory_utilization_pct=0.0,
            power_draw_w=0.0,
            power_limit_w=700.0,
            clock_speed_mhz=0,
            max_clock_mhz=2100,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=0,
            pcie_width=0,
            xid_errors=[79],  # XID 79 = GPU has fallen off the bus
            fan_speed_pct=0.0,
            status="critical",
            issues=[
                "GPU NOT RESPONDING — XID 79 (GPU has fallen off the bus)",
                "PCIe link down — check power cables and riser card",
                "Possible causes: PSU brownout, loose PCIe connector, hardware failure",
                "IMMEDIATE ACTION: Power cycle the node. If persists, reseat GPU.",
            ],
            timestamp=time.time(),
        ),
    },
    "nvlink_errors": {
        "weight": 0.05,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA H100 80GB HBM3 [GPU {gpu_id}]",
            temperature_c=round(random.uniform(50, 65), 1),
            utilization_pct=round(random.uniform(40, 60), 1),
            memory_used_gb=round(random.uniform(30, 50), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(37, 62), 1),
            power_draw_w=round(random.uniform(250, 400), 1),
            power_limit_w=700.0,
            clock_speed_mhz=random.randint(1800, 2100),
            max_clock_mhz=2100,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=5,
            pcie_width=16,
            xid_errors=[74, 74],  # XID 74 = NVLink error
            fan_speed_pct=round(random.uniform(40, 55), 1),
            status="warning",
            issues=[
                "NVLink errors detected (XID 74) — inter-GPU communication degraded",
                "Distributed training throughput may be reduced by 30-50%",
                "Check NVSwitch health and cable connections",
                "Run: nvidia-smi nvlink -s to check link status",
            ],
            timestamp=time.time(),
        ),
    },
    "clock_stuck": {
        "weight": 0.05,
        "generator": lambda gpu_id: GPUHealthReport(
            gpu_id=gpu_id,
            name=f"NVIDIA A100 80GB SXM4 [GPU {gpu_id}]",
            temperature_c=round(random.uniform(40, 50), 1),
            utilization_pct=round(random.uniform(80, 95), 1),
            memory_used_gb=round(random.uniform(40, 60), 1),
            memory_total_gb=80.0,
            memory_utilization_pct=round(random.uniform(50, 75), 1),
            power_draw_w=round(random.uniform(150, 200), 1),  # Low power = low clocks
            power_limit_w=400.0,
            clock_speed_mhz=210,  # Stuck at base
            max_clock_mhz=1410,
            ecc_single_bit_errors=0,
            ecc_double_bit_errors=0,
            pcie_gen=4,
            pcie_width=16,
            xid_errors=[],
            fan_speed_pct=round(random.uniform(30, 40), 1),
            status="warning",
            issues=[
                "GPU clock stuck at base frequency (210 MHz vs 1410 MHz max)",
                "Performance reduced by ~85% — likely power management issue",
                "Try: nvidia-smi -pm 1 && nvidia-smi -ac 1215,1410",
                "If persists, check nvidia-persistenced and power governor settings",
            ],
            timestamp=time.time(),
        ),
    },
}


def _pick_scenario() -> str:
    """Pick a mock scenario based on weighted probabilities."""
    scenarios = list(MOCK_SCENARIOS.keys())
    weights = [MOCK_SCENARIOS[s]["weight"] for s in scenarios]
    return random.choices(scenarios, weights=weights, k=1)[0]


def gpu_health_check(gpu_id: int | None = None) -> dict[str, Any]:
    """Check GPU health metrics.

    Args:
        gpu_id: Specific GPU index, or None for all GPUs (mock: 4 GPUs).

    Returns:
        Dict with GPU health reports and summary.
    """
    if GPU_MODE == "real":
        return _real_gpu_check(gpu_id)

    # Mock mode — simulate a 4-GPU node (DGX-like)
    num_gpus = 4
    if gpu_id is not None:
        if gpu_id < 0 or gpu_id >= num_gpus:
            return {"error": f"GPU {gpu_id} not found. Available: 0-{num_gpus - 1}"}
        scenario = _pick_scenario()
        report = MOCK_SCENARIOS[scenario]["generator"](gpu_id)
        return {"gpus": [asdict(report)], "summary": _summarize([report])}

    # Check all GPUs — most healthy, maybe one or two with issues
    reports = []
    for i in range(num_gpus):
        # First GPU gets a random scenario, rest are mostly healthy
        if i == 0:
            scenario = _pick_scenario()
        else:
            scenario = "healthy" if random.random() < 0.85 else _pick_scenario()
        reports.append(MOCK_SCENARIOS[scenario]["generator"](i))

    return {"gpus": [asdict(r) for r in reports], "summary": _summarize(reports)}


def _summarize(reports: list[GPUHealthReport]) -> dict[str, Any]:
    """Generate a summary of GPU health across the node."""
    critical = [r for r in reports if r.status == "critical"]
    warnings = [r for r in reports if r.status == "warning"]
    healthy = [r for r in reports if r.status == "healthy"]

    return {
        "total_gpus": len(reports),
        "healthy": len(healthy),
        "warnings": len(warnings),
        "critical": len(critical),
        "overall_status": (
            "critical" if critical else "warning" if warnings else "healthy"
        ),
        "action_required": len(critical) + len(warnings) > 0,
    }


def _real_gpu_check(gpu_id: int | None = None) -> dict[str, Any]:
    """Real GPU health check using pynvml. Requires NVIDIA GPU."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()

        if gpu_id is not None:
            if gpu_id >= count:
                return {"error": f"GPU {gpu_id} not found. Available: 0-{count - 1}"}
            indices = [gpu_id]
        else:
            indices = list(range(count))

        reports = []
        for idx in indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            max_clocks = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)

            issues = []
            status = "healthy"
            if temp > 83:
                issues.append(f"Temperature {temp}°C exceeds 83°C threshold")
                status = "warning"
            if mem.used / mem.total > 0.95:
                issues.append("Memory utilization >95%")
                status = "warning"

            report = GPUHealthReport(
                gpu_id=idx,
                name=name if isinstance(name, str) else name.decode(),
                temperature_c=float(temp),
                utilization_pct=float(util.gpu),
                memory_used_gb=round(mem.used / (1024**3), 1),
                memory_total_gb=round(mem.total / (1024**3), 1),
                memory_utilization_pct=round(mem.used / mem.total * 100, 1),
                power_draw_w=round(power, 1),
                power_limit_w=round(power_limit, 1),
                clock_speed_mhz=clocks,
                max_clock_mhz=max_clocks,
                ecc_single_bit_errors=0,
                ecc_double_bit_errors=0,
                pcie_gen=0,
                pcie_width=0,
                xid_errors=[],
                fan_speed_pct=0.0,
                status=status,
                issues=issues,
                timestamp=time.time(),
            )
            reports.append(report)

        pynvml.nvmlShutdown()
        return {"gpus": [asdict(r) for r in reports], "summary": _summarize(reports)}

    except ImportError:
        return {"error": "pynvml not installed. Install with: pip install pynvml"}
    except Exception as e:
        return {"error": f"GPU check failed: {e}"}
