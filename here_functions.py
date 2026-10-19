"""HERE routing and map attributes through the IFPEN Mobicloud service.

Dependency: requests (pip install requests).
Credentials: ~/.mobicloud/login.json with username and password fields.
"""

import json
import os
from pathlib import Path

import requests

__all__ = ["get_token", "get_route", "get_route_info"]

_BASE_URL = "https://mobicloud.ifpen.com"
_HERE_URL = f"{_BASE_URL}/geoservices/api/here"
_TIMEOUT = 30


def get_token(login_path=None) -> str:
    """Authenticate using the local login JSON and return its bearer token.

    login_path optionally overrides the default USERPROFILE/.mobicloud/login.json
    path. On systems without USERPROFILE, the user's home directory is used.
    File, JSON, missing-field and HTTP errors propagate to the caller.
    """
    if login_path is None:
        profile = Path(os.environ.get("USERPROFILE") or Path.home())
        login_path = profile / ".mobicloud" / "login.json"

    with Path(login_path).open(encoding="utf-8-sig") as login_file:
        login = json.load(login_file)

    response = requests.post(
        f"{_BASE_URL}/api/authenticate",
        json={
            "rememberMe": True,
            "username": login["username"],
            "password": login["password"],
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["id_token"]


def get_route(origin: str, destination: str, avoid_tolls: bool) -> dict:
    """Return a route response, using 'latitude,longitude' coordinate strings.

    Each call authenticates using get_token(). HTTP and JSON errors propagate.
    """
    token = get_token()
    params = {
        "origin": origin,
        "destination": destination,
        "transportMode": "car",
        "departureTime": "any",
        "routingMode": "fast",
        "alternatives": 0,
        "return": "polyline,summary,travelSummary,typicalDuration,elevation,tolls",
        "spans": (
            "names,length,duration,baseDuration,typicalDuration,functionalClass,"
            "segmentRef,maxSpeed,dynamicSpeedInfo,tollSystems,gates"
        ),
    }
    if avoid_tolls:
        params["avoid[features]"] = "tollRoad"

    response = requests.get(
        f"{_HERE_URL}/router.hereapi.com/v8/routes",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def get_route_info(route: dict, route_index: int = 0) -> list:
    """Return combined map segments for a response from get_route().

    Select the first route by default (or route_index), collect spans from all
    its sections in order, and request attributes in batches of 40 references.
    Singleton dictionaries are also accepted for routes, sections and spans.
    An empty routes or spans array returns an empty list. Missing fields and
    invalid route indexes raise their usual Python errors.
    """
    token = get_token()
    routes = route["routes"]
    if isinstance(routes, dict):
        routes = [routes]
    if not routes:
        return []

    sections = routes[route_index]["sections"]
    if isinstance(sections, dict):
        sections = [sections]

    segment_refs = []
    for section in sections:
        spans = section["spans"]
        if isinstance(spans, dict):
            spans = [spans]
        segment_refs.extend(span["segmentRef"] for span in spans)

    attributes = (
        "TRAFFIC_SIGN_FCn(CONDITION_TYPE,TRAFFIC_SIGN_TYPE),"
        "ADAS_ATTRIB_FCn(HPX,HPY,HPZ)"
    )
    segments = []
    for start in range(0, len(segment_refs), 40):
        response = requests.get(
            f"{_HERE_URL}/smap.hereapi.com/v8/maps/attributes/path",
            params={
                "transportMode": "car",
                "attributes": attributes,
                "segmentRefs": ";".join(segment_refs[start:start + 40]),
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        batch_segments = response.json()["segments"]
        if isinstance(batch_segments, dict):
            batch_segments = [batch_segments]
        segments.extend(batch_segments)

    return segments
