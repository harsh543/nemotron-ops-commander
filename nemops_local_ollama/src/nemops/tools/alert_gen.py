"""Alert generation tool — create structured OpsGenie-style alerts."""

import uuid
from datetime import datetime, timezone
from typing import Any


# Escalation policies keyed by severity
_ESCALATION_POLICIES = {
    "critical": {
        "policy": "gpu-critical-oncall",
        "notify": ["#gpu-incidents", "PagerDuty"],
        "response_sla_minutes": 15,
    },
    "warning": {
        "policy": "gpu-warning-review",
        "notify": ["#gpu-warnings"],
        "response_sla_minutes": 60,
    },
    "info": {
        "policy": "gpu-info-log",
        "notify": ["#gpu-monitoring"],
        "response_sla_minutes": 480,
    },
}


def generate_alert(
    severity: str,
    title: str,
    analysis: str,
    remediation_steps: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a structured alert with escalation metadata.

    Args:
        severity: One of critical, warning, info.
        title: Human-readable alert title.
        analysis: Root-cause or diagnostic summary.
        remediation_steps: Ordered list of actions.

    Returns:
        Dict with alert payload and dispatch status.
    """
    severity = severity.lower()
    if severity not in _ESCALATION_POLICIES:
        severity = "info"

    escalation = _ESCALATION_POLICIES[severity]

    alert_payload = {
        "alert_id": f"NEMOPS-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "title": title,
        "analysis": analysis,
        "remediation_steps": remediation_steps or [],
        "escalation": escalation,
        "source": "nemops-agent",
        "status": "open",
    }

    return {
        "alert": alert_payload,
        "dispatched": True,
        "message": (
            f"Alert {alert_payload['alert_id']} created with severity={severity}. "
            f"Escalation: {escalation['policy']} — SLA {escalation['response_sla_minutes']}m."
        ),
    }
