import pytest

from travel_planner.llm import (
    ApiKeyError,
    BadResponseError,
    BusyError,
    GeminiLLM,
    MissingKeyError,
    NetworkProblemError,
    QuotaExceededError,
    classify_error,
)
from travel_planner.models import BudgetAdvice


class FakeAPIError(Exception):
    def __init__(self, code, message="", status=""):
        super().__init__(f"{code} {status}. {message}")
        self.code, self.message, self.status = code, message, status


class ConnectError(Exception):  # same class name as httpx's
    pass


@pytest.mark.parametrize(
    "exc,kind",
    [
        (FakeAPIError(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"), "auth"),
        (FakeAPIError(401, "ACCESS_TOKEN_TYPE_UNSUPPORTED"), "auth"),
        (FakeAPIError(403, "PERMISSION_DENIED"), "auth"),
        (FakeAPIError(400, "User location is not supported for the API use."), "region"),
        (FakeAPIError(429, "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier"), "daily_quota"),
        (FakeAPIError(429, "You exceeded your current quota ... limit: 0, model: x"), "daily_quota"),
        (FakeAPIError(429, "Rate limited. Please retry in 12.5s"), "rate"),
        (FakeAPIError(404, "models/foo is not found for API version v1beta"), "model_missing"),
        (FakeAPIError(503, "The model is overloaded", "UNAVAILABLE"), "busy"),
        (FakeAPIError(400, "Something else bad"), "bad_request"),
        (ConnectError("boom"), "network"),
        (RuntimeError("???"), "unknown"),
    ],
)
def test_classify(exc, kind):
    assert classify_error(exc)[0] == kind


def test_rate_retry_delay_parsed():
    assert classify_error(FakeAPIError(429, "please retry in 23.4s")) == ("rate", 23.4)
    assert classify_error(FakeAPIError(429, "{'retryDelay': '17s'}"))[1] == 17.0


def make_llm(models=("m1", "m2"), script=None):
    """GeminiLLM whose network call is replaced by a script of outcomes."""
    sleeps = []
    llm = GeminiLLM("fake-key", list(models), sleep=sleeps.append)
    script = list(script or [])
    calls = []

    def fake_call(model, schema, system, prompt, temperature):
        calls.append(model)
        outcome = script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    llm._call = fake_call
    return llm, calls, sleeps


GOOD = '{"items":[{"category":"Food","amount":10,"notes":"x"}],"summary":"ok","money_saving_tips":["t"]}'


def test_missing_key():
    with pytest.raises(MissingKeyError):
        GeminiLLM("", ["m"])


def test_success_and_fenced_json():
    llm, calls, _ = make_llm(script=["```json\n" + GOOD + "\n```"])
    out = llm.generate(BudgetAdvice, "s", "p")
    assert out.summary == "ok" and calls == ["m1"] and llm.models_used == ["m1"]


def test_daily_quota_switches_model_and_remembers():
    quota = FakeAPIError(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    llm, calls, _ = make_llm(script=[quota, GOOD, GOOD])
    llm.generate(BudgetAdvice, "s", "p")
    llm.generate(BudgetAdvice, "s", "p")  # m1 must be skipped now
    assert calls == ["m1", "m2", "m2"]


def test_all_models_quota_exhausted():
    quota = FakeAPIError(429, "limit: 0")
    llm, _, _ = make_llm(script=[quota, quota])
    with pytest.raises(QuotaExceededError):
        llm.generate(BudgetAdvice, "s", "p")


def test_rate_limit_waits_then_succeeds():
    llm, calls, sleeps = make_llm(script=[FakeAPIError(429, "retry in 10s"), GOOD])
    llm.generate(BudgetAdvice, "s", "p")
    assert calls == ["m1", "m1"] and sleeps and 10 <= sleeps[0] <= 70


def test_rate_limit_forever_moves_on():
    rl = FakeAPIError(429, "retry in 1s")
    llm, calls, _ = make_llm(script=[rl, rl, rl, GOOD])
    llm.generate(BudgetAdvice, "s", "p")
    assert calls == ["m1", "m1", "m1", "m2"]


def test_bad_key_raises_friendly_error():
    llm, _, _ = make_llm(script=[FakeAPIError(400, "API key not valid")])
    with pytest.raises(ApiKeyError) as info:
        llm.generate(BudgetAdvice, "s", "p")
    assert "aistudio.google.com/apikey" in info.value.user_message


def test_bad_json_retries_then_gives_up():
    llm, calls, _ = make_llm(script=["not json", "{}", "[]"])
    with pytest.raises(BadResponseError):
        llm.generate(BudgetAdvice, "s", "p")
    assert len(calls) == 3


def test_bad_json_then_good():
    llm, _, _ = make_llm(script=["garbage", GOOD])
    assert llm.generate(BudgetAdvice, "s", "p").summary == "ok"


def test_busy_model_falls_back_to_next_model():
    busy = FakeAPIError(503, "high demand", "UNAVAILABLE")
    llm, calls, sleeps = make_llm(script=[busy, busy, busy, GOOD])
    assert llm.generate(BudgetAdvice, "s", "p").summary == "ok"
    assert calls == ["m1", "m1", "m1", "m2"] and len(sleeps) == 2 and llm.models_used == ["m2"]


def test_busy_everywhere_gives_friendly_busy_error():
    busy = FakeAPIError(503, "high demand", "UNAVAILABLE")
    llm, calls, _ = make_llm(script=[busy] * 6)
    with pytest.raises(BusyError) as info:
        llm.generate(BudgetAdvice, "s", "p")
    assert len(calls) == 6 and "503" in info.value.technical


def test_network_error():
    llm, _, _ = make_llm(script=[ConnectError("x")] * 3)
    with pytest.raises(NetworkProblemError):
        llm.generate(BudgetAdvice, "s", "p")
