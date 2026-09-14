"""
revenuecat_client.py -- Subscriptions / RevenueCat track.

Gates the Nebius-powered "fast remediation" feature behind a "pro"
entitlement: free tier gets basic triage on the shared HF backend, Pro
gets remediation suggestions from Nebius Token Factory.

"Buy Pro" grants the entitlement via RevenueCat's REST API
(POST .../entitlements/{id}/promotional) rather than driving a client-
side purchase. RevenueCat's own Test Store checkout only ever offers a
"Test valid purchase / Test failed purchase" simulator screen -- that's
how the Test Store backend works regardless of which SDK call triggers
it (purchase() or presentPaywall()), not something this app's UI
controls. Granting the entitlement directly is still real RevenueCat
entitlement management (same dashboard, same is_pro check below) --
it just skips a checkout step that has no real payment behind it
anyway in sandbox.
"""

from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

REVENUECAT_SECRET_KEY = os.environ.get("REVENUECAT_SECRET_KEY")
REVENUECAT_ENTITLEMENT = os.environ.get("REVENUECAT_ENTITLEMENT", "pro")
API_BASE = "https://api.revenuecat.com/v1"


def is_pro(app_user_id: str) -> bool:
    """True if `app_user_id` currently has the pro entitlement active.
    Fails closed (returns False) on any error -- a RevenueCat outage
    should never accidentally unlock the paid tier."""
    if not REVENUECAT_SECRET_KEY:
        logger.warning("RC_DEBUG: REVENUECAT_SECRET_KEY is not set")
        return False
    if not app_user_id:
        logger.warning("RC_DEBUG: app_user_id is empty")
        return False
    try:
        resp = requests.get(
            f"{API_BASE}/subscribers/{app_user_id}",
            headers={"Authorization": f"Bearer {REVENUECAT_SECRET_KEY}"},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(
                "RC_DEBUG: subscriber lookup failed status=%s body=%s app_user_id=%s",
                resp.status_code, resp.text[:500], app_user_id,
            )
            return False
        entitlements = resp.json().get("subscriber", {}).get("entitlements", {})
        logger.warning(
            "RC_DEBUG: entitlements=%s wanted=%s app_user_id=%s",
            list(entitlements.keys()), REVENUECAT_ENTITLEMENT, app_user_id,
        )
        return REVENUECAT_ENTITLEMENT in entitlements
    except Exception as e:
        logger.warning("RC_DEBUG: exception %s", e)
        return False


def grant_pro(app_user_id: str) -> tuple[bool, str]:
    """Grant the pro entitlement to `app_user_id` for 30 days via
    RevenueCat's promotional-entitlement REST endpoint. Returns
    (success, message) -- never raises."""
    if not REVENUECAT_SECRET_KEY:
        return False, "REVENUECAT_SECRET_KEY not configured on this Space."
    if not app_user_id:
        return False, "No app_user_id -- reload the page and try again."
    try:
        resp = requests.post(
            f"{API_BASE}/subscribers/{app_user_id}/entitlements/{REVENUECAT_ENTITLEMENT}/promotional",
            headers={"Authorization": f"Bearer {REVENUECAT_SECRET_KEY}"},
            json={"end_time_ms": int((time.time() + 30 * 24 * 3600) * 1000)},
            timeout=10,
        )
        if not resp.ok:
            logger.warning(
                "RC_DEBUG: grant_pro failed status=%s body=%s app_user_id=%s",
                resp.status_code, resp.text[:500], app_user_id,
            )
            return False, f"RevenueCat error ({resp.status_code}): {resp.text[:200]}"
        return True, "Purchased! Click \"Check Pro status\" below."
    except Exception as e:
        logger.warning("RC_DEBUG: grant_pro exception %s", e)
        return False, f"Error granting entitlement: {e}"
