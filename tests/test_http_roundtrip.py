"""Runs the REAL google-genai library against a tiny fake Gemini server on localhost.

This proves our request format (structured JSON schema, headers, model names) and
our parsing work with the real library, without needing internet or an API key.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from travel_planner.graph import plan_trip_request
from travel_planner.llm import ApiKeyError, GeminiLLM, QuotaExceededError
from tests.fakes import make_advice, make_plan, make_request


class FakeGemini(BaseHTTPRequestHandler):
    requests: list = []
    quota_models: set = set()
    bad_key = False

    def log_message(self, *a):  # silence
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = self.path.split("/models/")[1].split(":")[0]
        FakeGemini.requests.append({"model": model, "headers": dict(self.headers), "body": body, "path": self.path})

        if FakeGemini.bad_key:
            return self._reply(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key."}})
        if model in FakeGemini.quota_models:
            return self._reply(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                               "message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_model_per_day, limit: 20 GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})

        system = json.dumps(body.get("systemInstruction", body.get("system_instruction", {})))
        if "Request Analyst" in system:
            data = make_request().model_dump()
        elif "Trip Planner" in system:
            data = make_plan(3).model_dump()
        else:
            data = make_advice().model_dump()
        self._reply(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(data)}]},
                                          "finishReason": "STOP", "index": 0}]})


@pytest.fixture()
def server(monkeypatch):
    FakeGemini.requests, FakeGemini.quota_models, FakeGemini.bad_key = [], set(), False
    httpd = HTTPServer(("127.0.0.1", 0), FakeGemini)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{httpd.server_port}")
    yield FakeGemini
    httpd.shutdown()


def test_full_workflow_through_real_sdk(server):
    llm = GeminiLLM("AQ.fake-key-123", ["model-a"], sleep=lambda s: None)
    final = plan_trip_request(llm, "3 days in Goa for 2")
    assert final.request.destination == "Goa, India" and len(final.plan.days) == 3
    assert len(server.requests) == 3  # exactly 3 AI calls per trip

    first = server.requests[0]
    assert first["model"] == "model-a"
    assert first["headers"].get("x-goog-api-key") == "AQ.fake-key-123"
    cfg = first["body"]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert "responseSchema" in cfg or "responseJsonSchema" in cfg  # structured output is really requested


def test_daily_quota_falls_back_to_next_model(server):
    server.quota_models = {"model-a"}
    llm = GeminiLLM("k", ["model-a", "model-b"], sleep=lambda s: None)
    final = plan_trip_request(llm, "3 days in Goa")
    assert final.models_used == ["model-b"]
    models = [r["model"] for r in server.requests]
    assert models == ["model-a", "model-b", "model-b", "model-b"]  # model-a tried once, then remembered as exhausted


def test_all_quota_gone(server):
    server.quota_models = {"model-a", "model-b"}
    llm = GeminiLLM("k", ["model-a", "model-b"], sleep=lambda s: None)
    with pytest.raises(QuotaExceededError):
        plan_trip_request(llm, "3 days in Goa")


def test_invalid_key_message(server):
    server.bad_key = True
    llm = GeminiLLM("wrong", ["model-a"], sleep=lambda s: None)
    with pytest.raises(ApiKeyError):
        plan_trip_request(llm, "3 days in Goa")


def test_overloaded_model_503_falls_back_through_real_sdk(server, monkeypatch):
    """Reproduces the 'model is experiencing high demand' error you saw."""
    original = FakeGemini.do_POST

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path.split("/models/")[1].split(":")[0] == "model-a":
            FakeGemini.requests.append({"model": "model-a"})
            return self._reply(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                               "message": "This model is currently experiencing high demand."}})
        import io
        self.rfile = io.BytesIO(raw)
        return original(self)

    monkeypatch.setattr(FakeGemini, "do_POST", do_POST)
    llm = GeminiLLM("k", ["model-a", "model-b"], sleep=lambda s: None)
    final = plan_trip_request(llm, "3 days in Goa")
    assert final.models_used == ["model-b"] and len(final.plan.days) == 3
