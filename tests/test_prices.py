import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from travel_planner import usage
from travel_planner.graph import plan_trip_request
from travel_planner.models import LiveFlight, LiveHotel, LivePrices
from travel_planner.prices import (
    PriceCache, PriceError, fetch_live_prices, live_budget, make_fetcher, pick_hotel_price, trip_dates,
)
from tests.fakes import FakeLLM, make_request

FLIGHTS = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?x=1"},
    "best_flights": [{"flights": [{"airline": "IndiGo", "departure_airport": {"time": "2027-01-10 08:25"},
                                   "arrival_airport": {"time": "2027-01-10 09:35"}}],
                      "total_duration": 70, "price": 6500, "layovers": []}],
    "other_flights": [
        {"flights": [{"airline": "Air India"}, {"airline": "Air India"}], "total_duration": 240, "price": 5200,
         "layovers": [{"name": "BOM"}]},
        {"flights": [{"airline": "No price"}], "price": None},
    ],
}
HOTELS = {"properties": [
    {"name": "Pricey Resort", "rate_per_night": {"extracted_lowest": 9000}, "overall_rating": 4.6, "extracted_hotel_class": 5, "link": "https://h/1"},
    {"name": "Mid Hotel", "rate_per_night": {"extracted_lowest": 4000}, "total_rate": {"extracted_lowest": 12000}, "overall_rating": 4.1},
    {"name": "Cheap Stay", "rate_per_night": {"lowest": "₹1,800"}},
    {"name": "No Price Hotel"},
]}


class FakeSerp(BaseHTTPRequestHandler):
    hits: list = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        FakeSerp.hits.append(q)
        if FakeSerp.mode == "badkey":
            data, code = {"error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"}, 401
        elif FakeSerp.mode == "noquota":
            data, code = {"error": "Your account has run out of searches."}, 429
        elif FakeSerp.mode == "hotels_fail" and q["engine"] == "google_hotels":
            data, code = {"error": "Something broke"}, 500
        elif FakeSerp.mode == "noresults":
            data, code = {"error": "Google hasn't returned any results for this query."}, 200
        else:
            data, code = (FLIGHTS if q["engine"] == "google_flights" else HOTELS), 200
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def serp(monkeypatch):
    FakeSerp.hits, FakeSerp.mode = [], "ok"
    httpd = HTTPServer(("127.0.0.1", 0), FakeSerp)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("SERPAPI_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/search.json")
    monkeypatch.setattr(usage, "PRICE_COUNTER", usage.DailyCounter())
    import travel_planner.prices as P
    monkeypatch.setattr(P, "CACHE", PriceCache())
    yield FakeSerp
    httpd.shutdown()


TODAY = date(2026, 11, 12)


def req(**kw):
    return make_request(start_date="2027-01-10", end_date="2027-01-13", num_days=4, **kw)


def test_dates():
    s, e, sample = trip_dates(req(), TODAY)
    assert (s.isoformat(), e.isoformat(), sample) == ("2027-01-10", "2027-01-13", False)
    s, e, sample = trip_dates(make_request(start_date="2020-01-01", num_days=5), TODAY)  # past -> example dates
    assert sample and (s - TODAY).days == 30 and (e - s).days == 4
    s, e, _ = trip_dates(make_request(start_date="", num_days=1), TODAY)
    assert (e - s).days == 1


def test_parse_flights_and_hotels(serp):
    live = fetch_live_prices(req(), "KEY", today=TODAY)
    assert [f.price_per_person for f in live.flights] == [5200, 6500]       # sorted, per person, no-price skipped
    assert live.flights[0].airline == "Air India" and live.flights[0].stops == 1
    assert live.flights[1].departure.endswith("08:25") and live.flights[1].link.startswith("https://www.google")
    assert [h.name for h in live.hotels] == ["Cheap Stay", "Mid Hotel", "Pricey Resort"]  # sorted, no-price skipped
    assert live.hotels[0].price_per_night == 1800 and live.hotels[1].total_price == 12000
    assert not live.sample_dates and live.depart_date == "2027-01-10"
    fl = next(h for h in serp.hits if h["engine"] == "google_flights")
    assert (fl["departure_id"], fl["arrival_id"], fl["adults"], fl["type"], fl["return_date"]) == ("PNQ", "GOI", "1", "1", "2027-01-13")
    assert fl["api_key"] == "KEY" and fl["currency"] == "INR"
    ho = next(h for h in serp.hits if h["engine"] == "google_hotels")
    assert ho["q"] == "hotels in Goa, India" and ho["check_in_date"] == "2027-01-10" and ho["adults"] == "2"


def test_no_origin_means_hotels_only(serp):
    live = fetch_live_prices(req(origin="", origin_iata=""), "K", today=TODAY)
    assert live.hotels and not live.flights and any("Travelling from" in n for n in live.notes)
    assert all(h["engine"] == "google_hotels" for h in serp.hits)


def test_sample_dates_are_labelled(serp):
    live = fetch_live_prices(make_request(start_date="", num_days=3), "K", today=TODAY)
    assert live.sample_dates and any("example dates" in n for n in live.notes)


def test_one_side_failing_still_returns_the_other(serp):
    serp.mode = "hotels_fail"
    live = fetch_live_prices(req(), "K", today=TODAY)
    assert live.flights and not live.hotels and any("hotels could not be loaded" in n for n in live.notes)


@pytest.mark.parametrize("mode,text", [("badkey", "key was rejected"), ("noquota", "allowance is used up"),
                                       ("noresults", "No live flight or hotel prices")])
def test_errors_are_friendly(serp, mode, text):
    serp.mode = mode
    with pytest.raises(PriceError) as info:
        fetch_live_prices(req(), "K", today=TODAY)
    assert text in info.value.user_message


def test_unreachable_service(monkeypatch):
    monkeypatch.setenv("SERPAPI_BASE_URL", "http://127.0.0.1:9/search.json")  # nothing listens here
    with pytest.raises(PriceError) as info:
        fetch_live_prices(req(), "K", today=TODAY)
    assert "could not reach" in info.value.user_message


def test_fetcher_cache_and_daily_limit(serp):
    fetch = make_fetcher("K", daily_limit=1, today=TODAY)
    first = fetch(req())
    n = len(serp.hits)
    assert fetch(req()).flights == first.flights and len(serp.hits) == n     # cached: no new searches, no allowance used
    with pytest.raises(PriceError) as info:
        fetch(req(destination="Kochi, India", destination_iata="COK"))        # new search, allowance (1) already used
    assert "daily limit" in info.value.user_message


def test_failed_lookup_gives_the_allowance_back(serp):
    fetch = make_fetcher("K", daily_limit=1, today=TODAY)
    serp.mode = "badkey"
    with pytest.raises(PriceError):
        fetch(req())
    serp.mode = "ok"
    assert fetch(req()).flights


def test_budget_math():
    live = LivePrices(currency="INR", depart_date="2027-01-10", return_date="2027-01-13",
                      flights=[LiveFlight(airline="A", price_per_person=5000), LiveFlight(airline="B", price_per_person=7000)],
                      hotels=[LiveHotel(name="a", price_per_night=1000), LiveHotel(name="b", price_per_night=3000),
                              LiveHotel(name="c", price_per_night=9000)])
    r = make_request(travelers=3, travel_style="moderate", num_days=4)
    out = live_budget(live, r)
    assert out["Flights"][0] == 15000                       # cheapest 5000 x 3 travellers
    assert out["Accommodation"][0] == 3000 * 3 * 2          # median x 3 nights x 2 rooms (3 people)
    assert pick_hotel_price(live.hotels, "budget") == 1000 and pick_hotel_price(live.hotels, "luxury") == 9000
    assert pick_hotel_price([], "moderate") is None


# ---- inside the workflow -------------------------------------------------------------------------

def live_sample():
    return LivePrices(currency="INR", depart_date="2027-01-10", return_date="2027-01-13",
                      flights=[LiveFlight(airline="IndiGo", price_per_person=5000)],
                      hotels=[LiveHotel(name="Mid", price_per_night=2000)])


def test_live_prices_replace_ai_numbers_in_budget():
    llm, steps = FakeLLM(), []
    final = plan_trip_request(llm, "x", on_step=steps.append, price_fetcher=lambda r: live_sample())
    items = {i.category: i for i in final.budget.items}
    assert items["Flights"].amount == 10000 and "Live price" in items["Flights"].notes      # 5000 x 2 people
    assert items["Accommodation"].amount == 2000 * 3 * 1 and "Live price" in items["Accommodation"].notes
    assert final.budget.total == 10000 + 6000 + 4 * 10000                                  # 4 other AI lines of 10000
    assert final.live_prices.flights[0].airline == "IndiGo"
    assert "VERIFIED LIVE PRICES" in llm.calls[-1][1]
    assert "fetch_prices" in steps


def test_price_problems_never_break_the_plan():
    def boom(request):
        raise PriceError("quota gone")

    final = plan_trip_request(FakeLLM(), "x", price_fetcher=boom)
    assert final.live_prices is None and any("quota gone" in w for w in final.warnings)
    final = plan_trip_request(FakeLLM(), "x", price_fetcher=lambda r: 1 / 0)
    assert final.live_prices is None and any("unexpected problem" in w for w in final.warnings)


def test_without_fetcher_nothing_changes():
    steps = []
    final = plan_trip_request(FakeLLM(), "x", on_step=steps.append)
    assert final.live_prices is None and "fetch_prices" not in steps and final.budget.total == 60000.0


def test_old_saved_plans_still_load():
    from travel_planner.models import FinalPlan
    final = plan_trip_request(FakeLLM(), "x")
    data = json.loads(final.model_dump_json())
    for key in ("origin_iata", "destination_iata"):
        data["request"].pop(key)
    data.pop("live_prices")
    assert FinalPlan.model_validate(data).request.origin_iata == ""
