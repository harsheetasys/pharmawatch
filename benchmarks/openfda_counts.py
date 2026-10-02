"""Lesson 1: build a PRR 2x2 table for one drug-event pair from live FDA data."""
import math
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket

import urllib3.util.connection as urllib3_conn

# Skip IPv6: on this network IPv6 attempts hang ~20s each before falling back to IPv4.
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET
BASE = "https://api.fda.gov/drug/event.json"
API_KEY = os.getenv("OPENFDA_API_KEY")

# One shared connection that is re-used for every call (no new handshake each time),
# with automatic retries when FDA's server drops, times out, or says "slow down" (429).
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, connect=5, read=3, backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
)))


def total(search: str) -> int:
    """How many FAERS reports match this search?"""
    params = {"search": search, "limit": 1}
    if API_KEY:
        params["api_key"] = API_KEY
    r = SESSION.get(BASE, params=params, timeout=(10, 60))   # 10s to connect, 60s to answer
    if r.status_code == 404:          # openFDA says 404 when nothing matches
        return 0
    r.raise_for_status()
    return r.json()["meta"]["results"]["total"]

def two_by_two(drug: str, event: str, end_date: str = "20111231") -> dict:
    date = f"receivedate:[19680101 TO {end_date}]"
    drug_q = f'patient.drug.openfda.generic_name:"{drug}"'
    event_q = f'patient.reaction.reactionmeddrapt:"{event}"'

    a = total(f"{date} AND {drug_q} AND {event_q}")   # drug + event
    n_drug = total(f"{date} AND {drug_q}")             # all reports with drug
    n_event = total(f"{date} AND {event_q}")           # all reports with event
    n_all = total(date)                                # every report

    b = n_drug - a
    c = n_event - a
    d = n_all - n_drug - c
    return {"a": a, "b": b, "c": c, "d": d}


def prr(t: dict) -> tuple[float, float]:
    a, b, c, d = (t[k] + 0.5 for k in "abcd")          # +0.5 avoids dividing by zero
    value = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    lower_05 = math.exp(math.log(value) - 1.645 * se)  # "PRR05" = 5th percentile
    return value, lower_05


if __name__ == "__main__":
    pairs = [
        ("warfarin", "gastrointestinal haemorrhage"),   # well-known bleeding risk
        ("levothyroxine", "gastrointestinal haemorrhage"),
    ]
    for drug, event in pairs:
        t = two_by_two(drug, event)
        value, low = prr(t)
        print(f"{drug:15s} + {event}")
        print(f"   table: {t}")
        print(f"   PRR = {value:.2f}   PRR05 = {low:.2f}\n")