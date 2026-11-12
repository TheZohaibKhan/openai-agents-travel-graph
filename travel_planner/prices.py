"""Live flight and hotel prices from Google Flights / Google Hotels, through SerpApi.

Why SerpApi: Amadeus closed its free self-service API in July 2026. SerpApi has one key for both
flights and hotels and a small free plan. If anything goes wrong here the trip planner simply keeps
the AI estimates, so a price problem can never break a trip plan.
"""

from __future__ import annotations

import copy
import math
import os
import re
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import requests

from travel_planner import usage
from travel_planner.models import LiveFlight, LiveHotel, LivePrices, TripRequest

DEFAULT_BASE_URL = "https://serpapi.com/search.json"
TIMEOUT_SECONDS = 30
CACHE_SECONDS = 3600


class PriceError(Exception):
    """A problem getting live prices. `user_message` is safe to show."""

    def __init__(self, user_message: str, technical: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical = technical


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _num(value) -> float | None:
    """'₹12,345' / 12345 / '12345.5' -> float, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.]", "", value)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    return None


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def trip_dates(request: TripRequest, today: date | None = None) -> tuple[date, date, bool]:
    """Dates to price: (depart, return, are_they_just_examples)."""
    today = today or date.today()
    start, end = _parse_date(request.start_date), _parse_date(request.end_date)
    length = max(1, request.num_days - 1)
    if start and start > today:
        return start, (end if end and end > start else start + timedelta(days=length)), False
    sample_start = today + timedelta(days=30)
    return sample_start, sample_start + timedelta(days=length), True


def rooms_needed(travelers: int) -> int:
    return max(1, math.ceil(max(1, travelers) / 2))


def pick_hotel_price(hotels: list[LiveHotel], style: str) -> float | None:
    """A representative nightly price that matches the travel style."""
    prices = sorted(h.price_per_night for h in hotels if h.price_per_night > 0)
    if not prices:
        return None
    if style == "budget":
        return prices[0]
    if style == "luxury":
        return prices[-1]
    return float(statistics.median(prices))


def live_budget(live: LivePrices, request: TripRequest) -> dict[str, tuple[float, str]]:
    """Budget lines we can replace with real numbers: {category: (amount, note)}."""
    out: dict[str, tuple[float, str]] = {}
    if live.flights:
        cheapest = min(f.price_per_person for f in live.flights)
        out["Flights"] = (
            round(cheapest * request.travelers, 2),
            f"Live price: cheapest round trip found {cheapest:,.0f} {live.currency} per person x {request.travelers}",
        )
    nightly = pick_hotel_price(live.hotels, request.travel_style)
    if nightly:
        d1, d2 = _parse_date(live.depart_date), _parse_date(live.return_date)
        nights = max(1, (d2 - d1).days) if d1 and d2 else max(1, request.num_days - 1)
        rooms = rooms_needed(request.travelers)
        out["Accommodation"] = (
            round(nightly * nights * rooms, 2),
            f"Live price: {nightly:,.0f} {live.currency} per room per night x {nights} night(s) x {rooms} room(s)",
        )
    return out


# --------------------------------------------------------------------------
# talking to SerpApi
# --------------------------------------------------------------------------


def _get_json(http_get: Callable, params: dict) -> dict:
    base = os.getenv("SERPAPI_BASE_URL") or DEFAULT_BASE_URL
    try:
        response = http_get(base, params=params, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise PriceError("I could not reach the live-price service.", technical=str(exc)[:300]) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise PriceError("The live-price service sent an answer I could not read.",
                         technical=response.text[:300]) from exc
    if not isinstance(data, dict):
        raise PriceError("The live-price service sent an unexpected answer.", technical=str(data)[:300])

    error = str(data.get("error") or "")
    if error:
        low = error.lower()
        if "hasn't returned any results" in low or "no results" in low:
            return {}
        if "invalid api key" in low or "api key" in low and "invalid" in low:
            raise PriceError("The SerpApi key was rejected. Check SERPAPI_API_KEY in your secrets.", technical=error)
        if "run out of searches" in low or "out of searches" in low or "limit" in low:
            raise PriceError("The monthly live-price search allowance is used up.", technical=error)
        raise PriceError("The live-price service reported a problem.", technical=error[:300])
    if response.status_code >= 400:
        raise PriceError(f"The live-price service answered with error {response.status_code}.")
    return data


def fetch_flights(http_get: Callable, api_key: str, origin: str, dest: str, depart: date, back: date,
                  currency: str) -> list[LiveFlight]:
    """Cheapest round-trip options for ONE adult (we multiply by travellers ourselves)."""
    data = _get_json(http_get, {
        "engine": "google_flights", "api_key": api_key, "departure_id": origin.upper(), "arrival_id": dest.upper(),
        "outbound_date": depart.isoformat(), "return_date": back.isoformat(), "type": "1", "adults": "1",
        "currency": currency, "hl": "en",
    })
    link = str((data.get("search_metadata") or {}).get("google_flights_url") or "")
    flights: list[LiveFlight] = []
    for bucket in ("best_flights", "other_flights"):
        for item in data.get(bucket) or []:
            price = _num(item.get("price"))
            if not price or price <= 0:
                continue
            legs = item.get("flights") or []
            airlines = list(dict.fromkeys(str(leg.get("airline")) for leg in legs if leg.get("airline")))
            first = (legs[0].get("departure_airport") or {}).get("time", "") if legs else ""
            last = (legs[-1].get("arrival_airport") or {}).get("time", "") if legs else ""
            flights.append(LiveFlight(
                airline=", ".join(airlines) or "Airline not listed",
                price_per_person=price,
                duration_minutes=int(_num(item.get("total_duration")) or 0),
                stops=len(item.get("layovers") or []),
                departure=str(first), arrival=str(last), link=link,
            ))
    flights.sort(key=lambda f: f.price_per_person)
    return flights[:5]


def fetch_hotels(http_get: Callable, api_key: str, place: str, check_in: date, check_out: date,
                 currency: str) -> list[LiveHotel]:
    """Hotel options for a standard room for 2 adults."""
    data = _get_json(http_get, {
        "engine": "google_hotels", "api_key": api_key, "q": f"hotels in {place}",
        "check_in_date": check_in.isoformat(), "check_out_date": check_out.isoformat(),
        "adults": "2", "currency": currency, "hl": "en",
    })
    hotels: list[LiveHotel] = []
    for prop in data.get("properties") or []:
        nightly = _num((prop.get("rate_per_night") or {}).get("extracted_lowest")
                       or (prop.get("rate_per_night") or {}).get("lowest"))
        if not nightly or nightly <= 0 or not prop.get("name"):
            continue
        hotels.append(LiveHotel(
            name=str(prop["name"]),
            price_per_night=nightly,
            total_price=_num((prop.get("total_rate") or {}).get("extracted_lowest")
                             or (prop.get("total_rate") or {}).get("lowest")) or 0.0,
            rating=_num(prop.get("overall_rating")) or 0.0,
            stars=int(_num(prop.get("extracted_hotel_class")) or 0),
            link=str(prop.get("link") or ""),
        ))
        if len(hotels) >= 12:
            break
    hotels.sort(key=lambda h: h.price_per_night)
    return hotels[:6]


def fetch_live_prices(request: TripRequest, api_key: str, http_get: Callable | None = None,
                      today: date | None = None) -> LivePrices:
    """Look up flights and hotels in parallel. Raises PriceError only if nothing at all was found."""
    http_get = http_get or requests.get
    depart, back, sample = trip_dates(request, today)
    live = LivePrices(currency=request.currency, depart_date=depart.isoformat(), return_date=back.isoformat(),
                      sample_dates=sample)
    if sample:
        live.notes.append(
            f"No travel dates were given, so prices are for example dates ({depart.isoformat()} to "
            f"{back.isoformat()}). Add your real dates in 'Optional details' for accurate prices."
        )

    can_fly = bool(request.origin_iata and request.destination_iata
                   and request.origin_iata.upper() != request.destination_iata.upper())
    if not can_fly:
        live.notes.append("Flight prices need a 'Travelling from' city, so only hotels were looked up.")

    errors: list[PriceError] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        flight_job = pool.submit(fetch_flights, http_get, api_key, request.origin_iata, request.destination_iata,
                                 depart, back, request.currency) if can_fly else None
        hotel_job = pool.submit(fetch_hotels, http_get, api_key, request.destination, depart, back,
                                request.currency)
        for name, job in (("flights", flight_job), ("hotels", hotel_job)):
            if job is None:
                continue
            try:
                result = job.result()
            except PriceError as exc:
                errors.append(exc)
                live.notes.append(f"Live {name} could not be loaded ({exc.user_message})")
                continue
            if name == "flights":
                live.flights = result
            else:
                live.hotels = result

    if not live.flights and not live.hotels:
        if errors:
            raise errors[0]
        raise PriceError("No live flight or hotel prices were found for this trip.")
    if can_fly and not live.flights and not errors:
        live.notes.append("No live flights were found for these dates and airports.")
    return live


# --------------------------------------------------------------------------
# cache + daily allowance
# --------------------------------------------------------------------------


class PriceCache:
    def __init__(self, seconds: int = CACHE_SECONDS) -> None:
        self._seconds = seconds
        self._lock = threading.Lock()
        self._data: dict[tuple, tuple[float, LivePrices]] = {}

    def get(self, key: tuple) -> LivePrices | None:
        with self._lock:
            hit = self._data.get(key)
            if hit and time.time() - hit[0] < self._seconds:
                return copy.deepcopy(hit[1])
            self._data.pop(key, None)
            return None

    def put(self, key: tuple, value: LivePrices) -> None:
        with self._lock:
            self._data[key] = (time.time(), copy.deepcopy(value))


CACHE = PriceCache()


def make_fetcher(api_key: str, daily_limit: int = 0, http_get: Callable | None = None,
                 today: date | None = None) -> Callable[[TripRequest], LivePrices]:
    """The function the planner calls. Adds a 1-hour cache and a daily allowance for lookups."""

    def fetch(request: TripRequest) -> LivePrices:
        depart, back, _ = trip_dates(request, today)
        key = (request.origin_iata.upper(), request.destination_iata.upper(), request.destination.lower(),
               depart.isoformat(), back.isoformat(), request.currency)
        cached = CACHE.get(key)
        if cached:
            return cached
        if not usage.PRICE_COUNTER.try_take(daily_limit):
            raise PriceError("The daily limit for live-price lookups was reached (this protects the free search "
                             "allowance).")
        try:
            live = fetch_live_prices(request, api_key, http_get=http_get, today=today)
        except PriceError:
            usage.PRICE_COUNTER.give_back()
            raise
        CACHE.put(key, live)
        return live

    return fetch
