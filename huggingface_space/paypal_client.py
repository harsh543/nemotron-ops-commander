"""
paypal_client.py -- PayPal Orders v2 (sandbox) client for the Pro tier.

Ported from latch/paypal_client.py's pay_with_test_card() path: a card
payment_source captures a real sandbox order synchronously, server-to-server,
with no redirect and no buyer-facing approval step. That's what makes a
single "Buy Pro" button possible -- unlike RevenueCat's Test Store checkout,
there's no picker of test outcomes to click through.

Credentials come from PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET (sandbox app).
PAYPAL_BASE_URL defaults to the sandbox REST API -- do not point this at
api-m.paypal.com without a live app, live credentials, and a deliberate
decision to move real money.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import requests

PAYPAL_BASE_URL = os.environ.get("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")
TOKEN_URL = f"{PAYPAL_BASE_URL}/v1/oauth2/token"
ORDERS_URL = f"{PAYPAL_BASE_URL}/v2/checkout/orders"

# Demo safety ceiling -- enforced server-side regardless of the amount a
# caller asks for. Raise only for a deliberate, non-demo reason.
MAX_PAYMENT_AMOUNT = float(os.environ.get("MAX_PAYMENT_AMOUNT", "5.00"))

SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP"}

# PayPal's published sandbox test card -- Luhn-valid, sandbox-only, captures
# synchronously with no 3-D Secure challenge. Never use in a live app; this
# only works because PAYPAL_BASE_URL is the sandbox API.
TEST_CARD = {
    "number": "4111111111111111",
    "expiry": "2030-12",
    "security_code": "123",
    "name": "Sandbox Buyer",
}


class PayPalError(RuntimeError):
    """Raised when PayPal auth or Orders API calls fail."""


class PaymentAmountError(PayPalError):
    """Raised when a requested amount violates the demo safety ceiling."""


class PayPalAuth:
    """OAuth token manager: fetch once, cache, refresh 5 min before expiry."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def token(self) -> str:
        if self._token and time.time() < self._expires_at - 300:
            return self._token
        resp = requests.post(
            TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        if not resp.ok:
            raise PayPalError(f"OAuth token request failed ({resp.status_code}): {resp.text[:300]}")
        body = resp.json()
        self._token = body["access_token"]
        self._expires_at = time.time() + int(body.get("expires_in", 3600))
        return self._token


def pay_with_test_card(auth: PayPalAuth, amount: float, currency: str, description: str) -> dict:
    """Create AND capture a payment in one call using a card payment_source.

    Captures synchronously server-to-server -- no browser, no login, no
    picker of test outcomes. Returns {orderId, status, captureId, amount,
    currency}. Raises PayPalError if the capture doesn't complete."""
    if amount <= 0:
        raise PaymentAmountError("amount must be positive")
    if amount > MAX_PAYMENT_AMOUNT:
        raise PaymentAmountError(
            f"amount {amount:.2f} exceeds the demo ceiling of {MAX_PAYMENT_AMOUNT:.2f} {currency}"
        )
    currency = currency.upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise PayPalError(f"unsupported currency: {currency}")

    resp = requests.post(
        ORDERS_URL,
        headers={
            "Authorization": f"Bearer {auth.token()}",
            "Content-Type": "application/json",
            # Required whenever payment_source is set on order creation --
            # lets a retried request be recognized as a duplicate instead of
            # double-charging.
            "PayPal-Request-Id": str(uuid.uuid4()),
        },
        json={
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "description": description[:127],
                    "amount": {"currency_code": currency, "value": f"{amount:.2f}"},
                }
            ],
            "payment_source": {"card": TEST_CARD},
        },
        timeout=30,
    )
    if not resp.ok:
        raise PayPalError(f"pay_with_test_card failed ({resp.status_code}): {resp.text[:300]}")
    body = resp.json()
    if body["status"] != "COMPLETED":
        raise PayPalError(f"card payment did not complete synchronously: status={body['status']}")

    capture = body["purchase_units"][0]["payments"]["captures"][0]
    return {
        "orderId": body["id"],
        "status": body["status"],
        "captureId": capture["id"],
        "amount": capture["amount"]["value"],
        "currency": capture["amount"]["currency_code"],
    }
