"""
revenuecat_client.py -- Subscriptions / RevenueCat track.

Gates the Nebius-powered "fast remediation" feature behind a "pro"
entitlement: free tier gets basic triage on the shared HF backend, Pro
(unlocked via a RevenueCat Web Billing Test Store purchase, no real
money) gets remediation suggestions from Nebius Token Factory.

Entitlement checks are server-side REST calls (RevenueCat's secret key
never reaches the browser); the purchase itself happens client-side via
the Web Billing JS SDK, using the public key, in app.py's embedded HTML.
"""

from __future__ import annotations

import logging
import os

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
