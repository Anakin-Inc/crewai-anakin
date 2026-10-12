import json
from typing import Any
from unittest.mock import create_autospec

import httpx
import pytest
from anakin import Anakin
from anakin.errors import AnakinError
from anakin.models import Document, SearchResult
from crewai.tools import BaseTool

from crewai_anakin import (
    ALL_TOOLS,
    AnakinBaseTool,
    AnakinBrowserTaskTool,
    AnakinMonitorControlTool,
    AnakinScrapeTool,
    get_anakin_tools,
    tools,
)

# Every tool on Anakin's hosted MCP server (mcp.anakin.io) and its parameters.
HOSTED_PARAMS = {
    "scrape": {"url", "country", "use_browser", "generate_json", "force_fresh",
               "session_id", "session_name"},
    "search": {"prompt", "limit"},
    "map": {"url", "limit", "depth", "limit_per_level", "search", "include_subdomains",
            "include_external_links", "use_browser"},
    "crawl": {"url", "max_pages", "depth", "include_patterns", "exclude_patterns", "country",
              "use_browser", "session_id", "session_name"},
    "agentic_search": {"prompt", "schema", "use_browser"},
    "wire_discover": {"q", "limit"},
    "wire_catalog": {"slug"},
    "wire_read_action": {"action_id", "params", "credential_id", "identity_id"},
    "wire_write_action": {"action_id", "params", "credential_id", "identity_id"},
    "wire_identities": {"catalog_id"},
    "wire_login": {"catalog_slug", "params", "identity_name", "source_id", "source_ref"},
    "wire_build": {"website_url", "goal", "actions", "catalog_id", "country", "credential",
                   "visibility", "force"},
    "wire_build_status": {"id", "status", "limit", "page", "include_events"},
    "monitor_create": {"url", "interval_minutes", "scope", "watch_mode", "watch_format",
                       "output_schema", "ai_mode", "ai_goal", "alert_webhook_url",
                       "alert_emails", "use_browser", "country", "session_id", "max_pages",
                       "max_depth", "include_patterns", "exclude_patterns", "wire_action_id",
                       "wire_catalog_slug", "wire_params", "wire_credential_id",
                       "wire_watch_paths", "is_active", "expires_at"},
    "monitor_list": {"id"},
    "monitor_changes": {"id"},
    "monitor_control": {"id", "action"},
    "ai_visibility_search": {"query", "sources", "country", "include_full_content"},
    "ai_visibility_sources": set(),
    "session_list": {"domain"},
    "session_delete": {"id"},
    "browser_task": {"prompt", "url", "session_id", "output_schema", "max_steps", "timeout_ms"},
}  # fmt: skip
# Where this package names a hosted parameter differently.
RENAMED = {
    ("search", "prompt"): "query",
    ("agentic_search", "schema"): "output_schema",  # `schema` shadows a pydantic attribute
    ("wire_discover", "q"): "query",
    ("wire_build_status", "id"): "build_id",
    ("monitor_list", "id"): "monitor_id",
    ("monitor_changes", "id"): "monitor_id",
    ("monitor_control", "id"): "monitor_id",
    ("session_delete", "id"): "session_id",
}
# Hosted parameters the Python SDK has no option for.
NOT_IN_SDK = {("wire_build_status", "include_events")}
# Tools that change something; read_only=True leaves them out.
WRITE_TOOLS = {"wire_write_action", "wire_login", "wire_build", "monitor_create",
               "monitor_control", "session_delete", "browser_task"}  # fmt: skip


@pytest.fixture
def client() -> Any:
    # Autospec a real client, so a call that doesn't match the SDK's signature fails.
    return create_autospec(Anakin(api_key="ask_test"))


@pytest.fixture
def by_name(client: Any) -> dict[str, AnakinBaseTool]:
    return {t.name.removeprefix("anakin_"): t for t in get_anakin_tools("ask_test", client=client)}


def run(tool: AnakinBaseTool, **kwargs: Any) -> Any:
    return json.loads(tool.run(**kwargs))


def test_every_hosted_tool_is_a_crewai_tool(by_name: dict[str, AnakinBaseTool]) -> None:
    assert set(by_name) == set(HOSTED_PARAMS)
    assert len(ALL_TOOLS) == 22
    for tool in by_name.values():
        assert isinstance(tool, BaseTool)
        assert len(tool.description) > 40, tool.name
        tool.to_structured_tool()  # what a CrewAI agent actually gets


def test_tools_take_every_hosted_mcp_parameter(by_name: dict[str, AnakinBaseTool]) -> None:
    for name, hosted in HOSTED_PARAMS.items():
        taken = set(by_name[name].args_schema.model_fields)
        expected = {RENAMED.get((name, p), p) for p in hosted if (name, p) not in NOT_IN_SDK}
        assert expected <= taken, f"{name} is missing {sorted(expected - taken)}"


def test_read_only_drops_exactly_the_write_tools(client: Any) -> None:
    names = {
        t.name.removeprefix("anakin_") for t in get_anakin_tools(client=client, read_only=True)
    }
    assert names == set(HOSTED_PARAMS) - WRITE_TOOLS


def test_tools_share_one_client_and_hide_the_api_key(client: Any) -> None:
    all_tools = get_anakin_tools("ask_secret", client=client)
    assert {id(t.client) for t in all_tools} == {id(client)}
    # model_dump(), not model_dump_json(): crewai 1.0 can't JSON-serialize any BaseTool.
    dumped = all_tools[0].model_dump()
    assert "api_key" not in dumped
    assert "ask_secret" not in str(dumped)
    assert "ask_secret" not in repr(all_tools[0])


def test_a_tool_builds_its_own_client_when_used_alone() -> None:
    tool = AnakinScrapeTool(api_key="ask_test", base_url="https://example.test/v1")
    assert isinstance(tool.client, Anakin)
    assert tool.client is tool.client


def test_scrape_returns_the_page_as_json(by_name: dict[str, AnakinBaseTool], client: Any) -> None:
    client.scrape.return_value = Document.model_validate(
        {"id": "job_1", "url": "https://a.com/", "status": "completed",
         "markdown": "# A", "durationMs": 1200}
    )  # fmt: skip
    out = run(by_name["scrape"], url="https://a.com", use_browser=True)
    assert out["markdown"] == "# A"
    assert out["duration_ms"] == 1200
    assert "html" not in out  # None fields are dropped
    client.scrape.assert_called_once_with(
        "https://a.com", country="us", use_browser=True, generate_json=False,
        force_fresh=False, session_id=None, session_name=None,
    )  # fmt: skip


def test_search(by_name: dict[str, AnakinBaseTool], client: Any) -> None:
    client.search.return_value = SearchResult.model_validate(
        {"id": "s", "results": [{"url": "https://a.com", "title": "A"}]}
    )
    assert run(by_name["search"], query="crm pricing", limit=3) == [
        {"url": "https://a.com", "title": "A"}
    ]
    client.search.assert_called_once_with("crm pricing", limit=3)


@pytest.mark.parametrize(
    ("tool", "kwargs", "sdk_path", "expected"),
    [
        ("map", {"url": "https://a.com", "search": "blog"}, "map",
         (("https://a.com",), {"limit": 100, "depth": 2, "limit_per_level": 100, "search": "blog",
          "include_subdomains": False, "include_external_links": False, "use_browser": False})),
        ("crawl", {"url": "https://a.com", "include_patterns": ["/docs/*"]}, "crawl",
         (("https://a.com",), {"max_pages": 10, "depth": 1, "include_patterns": ["/docs/*"],
          "exclude_patterns": (), "country": "us", "use_browser": False, "session_id": None,
          "session_name": None})),
        ("agentic_search", {"prompt": "q", "output_schema": {"type": "object"}}, "agentic_search",
         (("q",), {"schema": {"type": "object"}, "use_browser": True})),
        ("wire_discover", {"query": "phones", "site": "walmart"}, "wire.discover",
         (("phones",), {"catalog": "walmart", "limit": 5})),
        ("wire_catalog", {}, "wire.catalogs", ((), {})),
        ("wire_catalog", {"slug": "amazon"}, "wire.catalog", (("amazon",), {})),
        ("wire_read_action", {"action_id": "amazon.search", "params": {"q": "tv"}}, "wire.run",
         (("amazon.search", {"q": "tv"}), {"credential_id": None, "identity_id": None})),
        ("wire_identities", {"catalog_id": "c1"}, "wire.identities", ((), {"catalog_id": "c1"})),
        ("wire_login", {"catalog_slug": "neb", "params": {"email": "e"}}, "wire.login",
         (("neb",), {"params": {"email": "e"}, "identity_name": None, "source_id": None,
          "source_ref": None})),
        ("wire_build", {"website_url": "https://x.com", "goal": "list jobs", "country": "US"},
         "wire.build",
         (("https://x.com", "list jobs"), {"actions": None, "catalog_id": None, "country": "US",
          "credential": None, "visibility": None, "force": False})),
        ("wire_build_status", {"build_id": "b1"}, "wire.get_build", (("b1",), {})),
        ("wire_build_status", {"status": "pending", "limit": 5, "page": 2}, "wire.builds",
         ((), {"status": "pending", "page": 2, "limit": 5})),
        ("monitor_list", {}, "monitors.list", ((), {})),
        ("monitor_list", {"monitor_id": "m1"}, "monitors.get", (("m1",), {})),
        ("monitor_changes", {"monitor_id": "m1"}, "monitors.changes", (("m1",), {})),
        ("ai_visibility_sources", {}, "ai_visibility.sources", ((), {})),
        ("session_list", {"domain": "amazon.com"}, "sessions.list", ((), {"domain": "amazon.com"})),
    ],
)  # fmt: skip
def test_tools_call_the_matching_sdk_method(
    by_name: dict[str, AnakinBaseTool],
    client: Any,
    tool: str,
    kwargs: dict[str, Any],
    sdk_path: str,
    expected: Any,
) -> None:
    target = client
    for part in sdk_path.split("."):
        target = getattr(target, part)
    target.return_value = {"ok": True}
    assert run(by_name[tool], **kwargs) == {"ok": True}
    args, kw = expected
    target.assert_called_once_with(*args, **kw)


def test_monitor_create_sends_only_the_options_given(
    by_name: dict[str, AnakinBaseTool], client: Any
) -> None:
    client.monitors.create.return_value = {"id": "m1"}
    run(
        by_name["monitor_create"], url="https://a.com", interval_minutes=60,
        watch_mode="specific_data", output_schema={"type": "object"}, ai_mode=True,
        ai_goal="price drops", watch_format="html", expires_at="2026-12-31",
    )  # fmt: skip
    client.monitors.create.assert_called_once_with(
        "https://a.com", 60, scope="page", watch_mode="specific_data", watch_format="html",
        output_schema={"type": "object"}, ai_mode=True, ai_goal="price drops",
        use_browser=False, expires_at="2026-12-31",
    )  # fmt: skip


def test_monitor_create_rejects_intervals_under_15_minutes(
    by_name: dict[str, AnakinBaseTool], client: Any
) -> None:
    with pytest.raises(ValueError, match="interval_minutes"):
        by_name["monitor_create"].run(url="https://a.com", interval_minutes=5)
    client.monitors.create.assert_not_called()


@pytest.mark.parametrize("action", ["pause", "resume", "run_now"])
def test_monitor_control_dispatches(
    by_name: dict[str, AnakinBaseTool], client: Any, action: str
) -> None:
    getattr(client.monitors, action).return_value = {"id": "m1"}
    assert run(by_name["monitor_control"], monitor_id="m1", action=action) == {"id": "m1"}
    getattr(client.monitors, action).assert_called_once_with("m1")


def test_monitor_control_delete_and_unknown_action(
    by_name: dict[str, AnakinBaseTool], client: Any
) -> None:
    assert run(by_name["monitor_control"], monitor_id="m1", action="delete") == {
        "id": "m1", "deleted": True
    }  # fmt: skip
    client.monitors.delete.assert_called_once_with("m1")
    with pytest.raises(ValueError):
        AnakinMonitorControlTool(client=client).run(monitor_id="m1", action="archive")


def test_session_delete(by_name: dict[str, AnakinBaseTool], client: Any) -> None:
    assert run(by_name["session_delete"], session_id="s1") == {"id": "s1", "deleted": True}
    client.sessions.delete.assert_called_once_with("s1")


def test_ai_visibility_drops_full_answers_unless_asked(
    by_name: dict[str, AnakinBaseTool], client: Any
) -> None:
    client.ai_visibility.search.return_value = {
        "results": [{"source": "chatgpt", "summary": "s", "full_content": "long"}]
    }
    assert (
        "full_content" not in run(by_name["ai_visibility_search"], query="best crm")["results"][0]
    )
    full = run(by_name["ai_visibility_search"], query="best crm", include_full_content=True)
    assert full["results"][0]["full_content"] == "long"


@pytest.mark.parametrize(
    ("action_id", "params"),
    [
        ("shop_checkout", {}),
        ("shop.place-order", {}),
        ("bank.send", {"note": "wire transfer to vendor"}),
        ("store.order", {"step": "buy now"}),
    ],
)
def test_write_actions_that_look_like_payments_are_refused(
    by_name: dict[str, AnakinBaseTool], client: Any, action_id: str, params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="payment"):
        by_name["wire_write_action"].run(action_id=action_id, params=params)
    client.wire.run.assert_not_called()


def test_ordinary_writes_and_all_reads_run(by_name: dict[str, AnakinBaseTool], client: Any) -> None:
    client.wire.run.return_value = {"status": "completed"}
    by_name["wire_write_action"].run(action_id="shop.add_to_cart", params={"sku": "1"})
    # Reads are never blocked, whatever their name.
    by_name["wire_read_action"].run(action_id="shop.checkout_page_prices")
    assert client.wire.run.call_count == 2


def test_financial_guard_can_be_turned_off(client: Any) -> None:
    client.wire.run.return_value = {"status": "completed"}
    allowed = {t.name: t for t in get_anakin_tools(client=client, allow_financial_actions=True)}
    allowed["anakin_wire_write_action"].run(action_id="shop_checkout")
    client.wire.run.assert_called_once()


def test_payment_builds_and_browser_tasks_are_refused(
    by_name: dict[str, AnakinBaseTool], client: Any
) -> None:
    with pytest.raises(ValueError, match="payment"):
        by_name["wire_build"].run(website_url="https://shop.test", goal="complete checkout")
    client.wire.build.assert_not_called()
    with pytest.raises(ValueError, match="payment"):
        by_name["browser_task"].run(prompt="buy now and pay with my saved card")


# ── browser_task talks to the REST API directly ───────────────────────────


def _mock_api(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    real_client = httpx.Client
    monkeypatch.setattr(
        tools.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setattr(tools.time, "sleep", lambda _: None)
    return seen


def test_browser_task_starts_then_polls_until_completed(
    by_name: dict[str, AnakinBaseTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _mock_api(
        monkeypatch,
        [
            httpx.Response(202, json={"workflow_id": "wf_1", "status": "queued"}),
            httpx.Response(200, json={"status": "running"}),
            httpx.Response(200, json={"status": "completed", "result": {"price": "$9"}}),
        ],
    )
    out = run(by_name["browser_task"], prompt="find the price", url="https://a.com", max_steps=10)
    assert out == {"price": "$9"}
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/ai/evaluate"
    assert seen[0].headers["X-API-Key"] == "ask_test"
    assert body == {"prompt": "find the price", "async": True, "url": "https://a.com",
                    "max_steps": 10}  # fmt: skip
    assert seen[1].url.path == "/v1/ai/jobs/wf_1"


def test_browser_task_failure_and_api_errors_raise(
    by_name: dict[str, AnakinBaseTool], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_api(
        monkeypatch,
        [
            httpx.Response(202, json={"workflow_id": "wf_1"}),
            httpx.Response(200, json={"status": "failed", "error": "page blocked"}),
            httpx.Response(401, json={"message": "Invalid API key"}),
        ],
    )
    with pytest.raises(AnakinError, match="page blocked"):
        by_name["browser_task"].run(prompt="do it")
    with pytest.raises(AnakinError, match="Invalid API key") as info:
        by_name["browser_task"].run(prompt="do it")
    assert info.value.status_code == 401


def test_browser_task_needs_a_key(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY", raising=False)
    with pytest.raises(AnakinError, match="No Anakin API key"):
        AnakinBrowserTaskTool(client=client).run(prompt="do it")


# ── A real CrewAI agent turn ──────────────────────────────────────────────


def test_a_crewai_agent_can_call_a_tool(by_name: dict[str, AnakinBaseTool], client: Any) -> None:
    client.search.return_value = SearchResult.model_validate(
        {"id": "s", "results": [{"url": "https://a.com", "title": "A"}]}
    )
    structured = by_name["search"].to_structured_tool()
    output = structured.invoke({"query": "crm pricing"})
    assert json.loads(output)[0]["url"] == "https://a.com"
    assert "query" in structured.args_schema.model_json_schema()["properties"]
