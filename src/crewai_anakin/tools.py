"""CrewAI tools for Anakin: the full Anakin API as tools for CrewAI agents."""

import json
import os
import re
import time
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import httpx
from anakin import Anakin
from anakin.errors import AnakinError
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr

DEFAULT_BASE_URL = "https://api.anakin.io/v1"

# Same guard as Anakin's hosted MCP server: refuse Wire writes, Wire builds and
# browser tasks that look like they complete a payment or move money. Targets
# payment completion (pay, checkout, purchase, place order, transfers), not
# "add to cart".
_FINANCIAL = re.compile(
    r"\b(payments?|pay\s?now|checkout|purchase|place\s?order|buy\s?now|wire\s?transfer"
    r"|remit(?:tance)?|payout|charge\s?card|transfer\s?funds)\b",
    re.IGNORECASE,
)

BROWSER_TASK_POLL_SECONDS = 3.0
BROWSER_TASK_MAX_SECONDS = 360.0


def _jsonable(value: Any) -> Any:
    """SDK models (pydantic) and containers of them, as plain JSON-ready data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=False, exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _looks_financial(*parts: Any) -> bool:
    text = " ".join(json.dumps(p, default=str) for p in parts if p is not None)
    # \b does not split snake_case or kebab-case ids such as "shop_checkout".
    return bool(_FINANCIAL.search(re.sub(r"[_-]+", " ", text)))


def _options(**values: Any) -> dict[str, Any]:
    """Keyword arguments with the unset (None) ones left out."""
    return {k: v for k, v in values.items() if v is not None}


class AnakinBaseTool(BaseTool):
    """
    Shared setup for the Anakin tools: the API key, one Anakin SDK client and
    JSON results. CrewAI hands a tool's result to the agent as text, so every
    tool returns JSON.
    """

    # Set on tools that change something on a website or in the user's Anakin
    # account, or start a long paid job. `get_anakin_tools(read_only=True)`
    # leaves them out.
    writes: ClassVar[bool] = False

    api_key: str | None = Field(default=None, exclude=True, repr=False)
    base_url: str = DEFAULT_BASE_URL
    poll_timeout: float = 600.0
    allow_financial_actions: bool = False
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ANAKIN_API_KEY",
                description="Anakin API key. Get one free at https://anakin.io/dashboard.",
                required=True,
            )
        ]
    )

    _client: Anakin | None = PrivateAttr(default=None)

    def __init__(self, client: Anakin | None = None, **data: Any) -> None:
        """
        Args:
            client: An `anakin.Anakin` client to use. By default each tool makes
                its own; `get_anakin_tools()` gives every tool one shared client.
            **data: Tool settings: api_key (default: the ANAKIN_API_KEY
                environment variable), base_url, poll_timeout,
                allow_financial_actions, and any CrewAI BaseTool field.
        """
        super().__init__(**data)
        self._client = client

    @property
    def client(self) -> Anakin:
        if self._client is None:
            self._client = Anakin(
                api_key=self.api_key, base_url=self.base_url, poll_timeout=self.poll_timeout
            )
        return self._client

    def _run(self, **kwargs: Any) -> str:
        args = self.args_schema.model_validate(kwargs)
        return json.dumps(_jsonable(self._call(args)), ensure_ascii=False)

    def _call(self, args: Any) -> Any:
        raise NotImplementedError

    def _refuse_if_financial(self, kind: str, *parts: Any) -> None:
        if not self.allow_financial_actions and _looks_financial(*parts):
            raise ValueError(
                f"Refused: this {kind} looks like a payment or money transfer. Complete "
                "payments directly on the site, or create the tool with "
                "allow_financial_actions=True."
            )


# ── Web data ──────────────────────────────────────────────────────────────


class ScrapeInput(BaseModel):
    url: str = Field(description="The page to fetch (http or https).")
    country: str = Field("us", description="Two-letter country to fetch from.")
    use_browser: bool = Field(
        False,
        description="Render with a headless browser. Only for JavaScript-heavy pages; "
        "slower and costs more.",
    )
    generate_json: bool = Field(False, description="Also extract structured fields with AI.")
    force_fresh: bool = Field(False, description="Skip the 24-hour cache.")
    session_id: str | None = Field(
        None,
        description="Saved browser session (from anakin_session_list) for pages behind a "
        "login. Use with use_browser.",
    )
    session_name: str | None = Field(
        None, description="A saved browser session's name, instead of session_id."
    )


class AnakinScrapeTool(AnakinBaseTool):
    name: str = "anakin_scrape"
    description: str = (
        "Fetch one web page and return its content as clean markdown. For a single "
        "known URL. For well-known sites (Amazon, LinkedIn, Zillow, ...) check "
        "anakin_wire_discover first: a Wire action returns structured data directly."
    )
    args_schema: type[BaseModel] = ScrapeInput

    def _call(self, args: ScrapeInput) -> Any:
        return self.client.scrape(
            args.url,
            country=args.country,
            use_browser=args.use_browser,
            generate_json=args.generate_json,
            force_fresh=args.force_fresh,
            session_id=args.session_id,
            session_name=args.session_name,
        )


class SearchInput(BaseModel):
    query: str = Field(description="What to search for, in natural language.")
    limit: int = Field(5, ge=1, le=20, description="Number of results (1-20).")


class AnakinSearchTool(AnakinBaseTool):
    name: str = "anakin_search"
    description: str = (
        "Search the web and return result URLs, titles and snippets. Returns "
        "immediately; scrape the results worth reading in full."
    )
    args_schema: type[BaseModel] = SearchInput

    def _call(self, args: SearchInput) -> Any:
        return self.client.search(args.query, limit=args.limit).results


class MapInput(BaseModel):
    url: str = Field(description="The site or section to start from.")
    limit: int = Field(100, description="Maximum URLs to return (up to 1000).")
    depth: int = Field(2, description="Link hops to follow from the start URL (up to 5).")
    limit_per_level: int = Field(
        100, description="Maximum URLs collected per link hop (up to 1000)."
    )
    search: str | None = Field(
        None, description="Only return URLs whose path or title matches this keyword."
    )
    include_subdomains: bool = Field(False, description="Also follow links to subdomains.")
    include_external_links: bool = Field(False, description="Also return links to other sites.")
    use_browser: bool = Field(
        False, description="Render pages with a headless browser (for single-page apps)."
    )


class AnakinMapTool(AnakinBaseTool):
    name: str = "anakin_map"
    description: str = (
        "List the URLs on a website. Use before anakin_crawl to decide which pages matter."
    )
    args_schema: type[BaseModel] = MapInput

    def _call(self, args: MapInput) -> Any:
        return self.client.map(
            args.url,
            limit=args.limit,
            depth=args.depth,
            limit_per_level=args.limit_per_level,
            search=args.search,
            include_subdomains=args.include_subdomains,
            include_external_links=args.include_external_links,
            use_browser=args.use_browser,
        )


class CrawlInput(BaseModel):
    url: str = Field(description="Where to start.")
    max_pages: int = Field(
        10, description="Hard cap on pages fetched (up to 500); the main cost lever."
    )
    depth: int = Field(1, description="Link hops to follow (up to 5).")
    include_patterns: list[str] | None = Field(
        None, description="Only crawl URLs matching these glob or regex patterns."
    )
    exclude_patterns: list[str] | None = Field(
        None, description="Skip URLs matching these patterns."
    )
    country: str = Field("us", description="Two-letter country to fetch from.")
    use_browser: bool = Field(False, description="Render with a headless browser.")
    session_id: str | None = Field(
        None, description="Saved browser session for pages behind a login."
    )
    session_name: str | None = Field(
        None, description="A saved browser session's name, instead of session_id."
    )


class AnakinCrawlTool(AnakinBaseTool):
    name: str = "anakin_crawl"
    description: str = (
        "Fetch many pages of one site as markdown, each with its own status. Scope it "
        "with include_patterns; an unscoped crawl spends credits on navigation and "
        "archive pages."
    )
    args_schema: type[BaseModel] = CrawlInput

    def _call(self, args: CrawlInput) -> Any:
        return self.client.crawl(
            args.url,
            max_pages=args.max_pages,
            depth=args.depth,
            include_patterns=args.include_patterns or (),
            exclude_patterns=args.exclude_patterns or (),
            country=args.country,
            use_browser=args.use_browser,
            session_id=args.session_id,
            session_name=args.session_name,
        )


class AgenticSearchInput(BaseModel):
    prompt: str = Field(description="The research question, as specific as possible.")
    output_schema: dict[str, Any] | None = Field(
        None, description="Optional JSON Schema for the structured result."
    )
    use_browser: bool = Field(True, description="Render cited pages with a headless browser.")


class AnakinAgenticSearchTool(AnakinBaseTool):
    name: str = "anakin_agentic_search"
    description: str = (
        "Research a question across many sources: searches, reads the citations and "
        "returns a summary plus structured data. Takes 1-5 minutes. Use when one page "
        "or a list of links can't answer the question (comparisons, market research)."
    )
    args_schema: type[BaseModel] = AgenticSearchInput

    def _call(self, args: AgenticSearchInput) -> Any:
        return self.client.agentic_search(
            args.prompt, schema=args.output_schema, use_browser=args.use_browser
        )


# ── Wire: pre-built site actions ──────────────────────────────────────────


class WireDiscoverInput(BaseModel):
    query: str = Field(
        description='The task, e.g. "top phones on walmart" or "a linkedin profile\'s '
        'work history".'
    )
    site: str | None = Field(None, description='Limit to one catalog slug, e.g. "amazon".')
    limit: int = Field(5, description="Number of matches.")


class AnakinWireDiscoverTool(AnakinBaseTool):
    name: str = "anakin_wire_discover"
    description: str = (
        "Find pre-built Wire actions for a task, ranked by relevance. Wire covers "
        "hundreds of popular sites with tested actions that return structured data, "
        "faster and cheaper than scraping or a browser task. Each match has its "
        "action_id, parameters, type (read or write), auth needs and credit cost."
    )
    args_schema: type[BaseModel] = WireDiscoverInput

    def _call(self, args: WireDiscoverInput) -> Any:
        return self.client.wire.discover(args.query, catalog=args.site, limit=args.limit)


class WireCatalogInput(BaseModel):
    slug: str | None = Field(None, description="Catalog slug to inspect. Omit to list all sites.")


class AnakinWireCatalogTool(AnakinBaseTool):
    name: str = "anakin_wire_catalog"
    description: str = (
        "Browse the Wire catalog. Without a slug, lists every supported site; with one "
        '(e.g. "walmart"), lists that site\'s actions with exact parameter schemas, '
        "type (read/write), auth mode, credit cost and login fields."
    )
    args_schema: type[BaseModel] = WireCatalogInput

    def _call(self, args: WireCatalogInput) -> Any:
        if args.slug:
            return self.client.wire.catalog(args.slug)
        return self.client.wire.catalogs()


class WireActionInput(BaseModel):
    action_id: str = Field(description="The action to run, from anakin_wire_discover.")
    params: dict[str, Any] | None = Field(
        None, description="Parameters matching the action's schema."
    )
    credential_id: str | None = Field(
        None,
        description="Credential for actions that need a login (from anakin_wire_identities).",
    )
    identity_id: str | None = Field(
        None, description="Alternative to credential_id: an identity whose credential to use."
    )


class AnakinWireReadActionTool(AnakinBaseTool):
    name: str = "anakin_wire_read_action"
    description: str = (
        "Run a Wire READ action: extracts data and changes nothing on the site (search "
        "listings, product details, profiles, dashboard data). Confirm the action's "
        'type is "read" with anakin_wire_discover or anakin_wire_catalog first.'
    )
    args_schema: type[BaseModel] = WireActionInput

    def _call(self, args: WireActionInput) -> Any:
        return self.client.wire.run(
            args.action_id,
            args.params or {},
            credential_id=args.credential_id,
            identity_id=args.identity_id,
        )


class AnakinWireWriteActionTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_wire_write_action"
    description: str = (
        "Run a Wire WRITE action: changes state on the site (submits a form, adds to a "
        "cart, posts content, updates settings). Confirm with the user before calling. "
        "Payments and money transfers are refused."
    )
    args_schema: type[BaseModel] = WireActionInput

    def _call(self, args: WireActionInput) -> Any:
        self._refuse_if_financial("action", args.action_id, args.params)
        return self.client.wire.run(
            args.action_id,
            args.params or {},
            credential_id=args.credential_id,
            identity_id=args.identity_id,
        )


class WireIdentitiesInput(BaseModel):
    catalog_id: str | None = Field(None, description="Only identities for this catalog.")


class AnakinWireIdentitiesTool(AnakinBaseTool):
    name: str = "anakin_wire_identities"
    description: str = (
        "List saved Wire identities and their credentials. A credential's id is the "
        'credential_id for Wire actions that need a login; check its status is "active".'
    )
    args_schema: type[BaseModel] = WireIdentitiesInput

    def _call(self, args: WireIdentitiesInput) -> Any:
        return self.client.wire.identities(catalog_id=args.catalog_id)


class WireLoginInput(BaseModel):
    catalog_slug: str = Field(description="The site's catalog slug.")
    params: dict[str, Any] | None = Field(
        None,
        description="Login fields from the catalog's login_input_schema (e.g. email, password).",
    )
    identity_name: str | None = Field(
        None, description="Name for the identity (required with source_id)."
    )
    source_id: str | None = Field(
        None, description="A connected 1Password identity source, instead of params."
    )
    source_ref: dict[str, Any] | None = Field(
        None, description="The 1Password item locator {vault_id, item_id, fields}."
    )


class AnakinWireLoginTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_wire_login"
    description: str = (
        "Sign in to a site for the user and get a credential_id usable right away. Only "
        "sign in to accounts the user owns, with details the user provided; never invent "
        "credentials. The password is not stored, only an encrypted session."
    )
    args_schema: type[BaseModel] = WireLoginInput

    def _call(self, args: WireLoginInput) -> Any:
        return self.client.wire.login(
            args.catalog_slug,
            params=args.params,
            identity_name=args.identity_name,
            source_id=args.source_id,
            source_ref=args.source_ref,
        )


class WireBuildInput(BaseModel):
    website_url: str = Field(description="The site to build for.")
    goal: str = Field(description="What the action should extract or do, specifically.")
    actions: list[str] | None = Field(
        None, description="Separate capabilities to build as separate actions."
    )
    catalog_id: str | None = Field(
        None, description="Add to an existing catalog instead of creating one."
    )
    country: str | None = Field(None, description="Two-letter country for geo-gated sites.")
    credential: dict[str, Any] | None = Field(
        None,
        description='Login for actions behind a sign-in: {"type": "plain", "username": ..., '
        '"password": ...}. Used once, not stored.',
    )
    visibility: Literal["private", "public"] | None = Field(
        None, description='"private" (default) or "public".'
    )
    force: bool = Field(False, description="Build even if similar actions exist.")


class AnakinWireBuildTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_wire_build"
    description: str = (
        "Request new Wire actions for a site not yet in the catalog. Wire builds and "
        "tests them; track progress with anakin_wire_build_status. Charges credits "
        "(refunded if the build fails); a login build costs much more, so confirm with "
        "the user first. Only use after anakin_wire_discover and anakin_wire_catalog "
        "find nothing."
    )
    args_schema: type[BaseModel] = WireBuildInput

    def _call(self, args: WireBuildInput) -> Any:
        self._refuse_if_financial("build", args.goal, args.website_url, args.actions)
        return self.client.wire.build(
            args.website_url,
            args.goal,
            actions=args.actions,
            catalog_id=args.catalog_id,
            country=args.country,
            credential=args.credential,
            visibility=args.visibility,
            force=args.force,
        )


class WireBuildStatusInput(BaseModel):
    build_id: str | None = Field(None, description="The id from anakin_wire_build's response.")
    status: str | None = Field(None, description="List mode only: filter by status.")
    limit: int | None = Field(None, description="List mode only: maximum builds per page.")
    page: int | None = Field(
        None, description="List mode only: 1-based page, to see builds past the first page."
    )


class AnakinWireBuildStatusTool(AnakinBaseTool):
    name: str = "anakin_wire_build_status"
    description: str = (
        "Check Wire builds. With build_id, returns that build: its status (poll about "
        'every 30 seconds while "pending" or "processing"), the published actions and '
        "a skipped list of anything it could not deliver. Without build_id, lists "
        "recent builds."
    )
    args_schema: type[BaseModel] = WireBuildStatusInput

    def _call(self, args: WireBuildStatusInput) -> Any:
        if args.build_id:
            return self.client.wire.get_build(args.build_id)
        return self.client.wire.builds(status=args.status, page=args.page, limit=args.limit)


# ── Monitors ──────────────────────────────────────────────────────────────


class MonitorCreateInput(BaseModel):
    url: str = Field(description="The page or site to watch (for wire scope, the site's URL).")
    interval_minutes: int = Field(ge=15, description="Minutes between checks (minimum 15).")
    scope: Literal["page", "site", "wire"] = Field(
        "page", description='"page", "site" (tracks pages added, removed, changed) or "wire".'
    )
    watch_mode: Literal["full_page", "specific_data"] = Field(
        "full_page",
        description='"full_page", or "specific_data" to track only output_schema fields.',
    )
    watch_format: Literal["markdown", "html", "cleaned_html"] | None = Field(
        None, description='full_page mode: compare "markdown" (default), "html" or "cleaned_html".'
    )
    output_schema: dict[str, Any] | None = Field(
        None, description="JSON Schema of fields to track (required for specific_data)."
    )
    ai_mode: bool = Field(False, description="Filter out trivial changes and summarize real ones.")
    ai_goal: str | None = Field(
        None, description='With ai_mode, which changes count, e.g. "only price drops".'
    )
    alert_webhook_url: str | None = Field(None, description="Webhook to notify on change.")
    alert_emails: list[str] | None = Field(None, description="Emails to notify on change.")
    use_browser: bool = Field(False, description="Render with a headless browser.")
    country: str | None = Field(None, description="Two-letter country to check from.")
    session_id: str | None = Field(
        None, description="Saved browser session for pages behind a login."
    )
    max_pages: int | None = Field(None, description="Site scope: pages per check.")
    max_depth: int | None = Field(
        None, description="Site scope: link depth to crawl (1-5, default 2)."
    )
    include_patterns: list[str] | None = Field(None, description="Site scope: only these URLs.")
    exclude_patterns: list[str] | None = Field(None, description="Site scope: skip these URLs.")
    wire_action_id: str | None = Field(
        None, description="Wire scope: the action to run each check."
    )
    wire_catalog_slug: str | None = Field(
        None, description="Wire scope: the action's catalog slug."
    )
    wire_params: dict[str, Any] | None = Field(
        None, description="Wire scope: the action's parameters."
    )
    wire_credential_id: str | None = Field(
        None, description="Wire scope: credential for actions behind a login."
    )
    wire_watch_paths: list[str] | None = Field(
        None, description="Wire scope: JSON paths to compare instead of the whole result."
    )
    is_active: bool | None = Field(None, description="False creates the monitor paused.")
    expires_at: str | None = Field(
        None, description="Date the monitor pauses itself (ISO 8601 or YYYY-MM-DD)."
    )


class AnakinMonitorCreateTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_monitor_create"
    description: str = (
        "Watch a page, a whole site or a Wire action on a schedule and record changes, "
        "optionally alerting a webhook or email. Every check costs credits: full_page "
        "2, specific_data 3, ai_mode +1. Tell the user the credits per day "
        "((1440 / interval_minutes) x cost per check) before creating."
    )
    args_schema: type[BaseModel] = MonitorCreateInput

    def _call(self, args: MonitorCreateInput) -> Any:
        options = args.model_dump(exclude={"url", "interval_minutes"}, exclude_none=True)
        return self.client.monitors.create(args.url, args.interval_minutes, **options)


class MonitorIdInput(BaseModel):
    monitor_id: str = Field(
        description="The monitor (from anakin_monitor_list or anakin_monitor_create)."
    )


class MonitorListInput(BaseModel):
    monitor_id: str | None = Field(None, description="Return just this monitor.")


class AnakinMonitorListTool(AnakinBaseTool):
    name: str = "anakin_monitor_list"
    description: str = (
        "List the user's monitors, or with monitor_id, one monitor's settings and status "
        "(next and last check, active state, cost per check, alerts)."
    )
    args_schema: type[BaseModel] = MonitorListInput

    def _call(self, args: MonitorListInput) -> Any:
        if args.monitor_id:
            return self.client.monitors.get(args.monitor_id)
        return self.client.monitors.list()


class AnakinMonitorChangesTool(AnakinBaseTool):
    name: str = "anakin_monitor_changes"
    description: str = "Get the changes a monitor detected, each with its diff or summary."
    args_schema: type[BaseModel] = MonitorIdInput

    def _call(self, args: MonitorIdInput) -> Any:
        return self.client.monitors.changes(args.monitor_id)


class MonitorControlInput(BaseModel):
    monitor_id: str = Field(description="The monitor.")
    action: Literal["pause", "resume", "run_now", "delete"] = Field(
        description='"pause", "resume", "run_now" or "delete".'
    )


class AnakinMonitorControlTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_monitor_control"
    description: str = (
        "Pause, resume, run now or delete a monitor. run_now is billed like a check. "
        "delete permanently removes the monitor and its history: confirm with the user "
        "first."
    )
    args_schema: type[BaseModel] = MonitorControlInput

    def _call(self, args: MonitorControlInput) -> Any:
        monitors = self.client.monitors
        if args.action == "pause":
            return monitors.pause(args.monitor_id)
        if args.action == "resume":
            return monitors.resume(args.monitor_id)
        if args.action == "run_now":
            return monitors.run_now(args.monitor_id)
        monitors.delete(args.monitor_id)
        return {"id": args.monitor_id, "deleted": True}


# ── AI Visibility ─────────────────────────────────────────────────────────


class AiVisibilitySearchInput(BaseModel):
    query: str = Field(max_length=2000, description="The question to ask every engine.")
    sources: list[str] | None = Field(
        None,
        description="Engine slugs to ask (see anakin_ai_visibility_sources). Default: all enabled.",
    )
    country: str = Field("us", description="Two-letter country for geo-specific answers.")
    include_full_content: bool = Field(
        False,
        description="Also return each engine's full answer, which is long. Summaries and "
        "the synthesis are always returned.",
    )


class AnakinAiVisibilitySearchTool(AnakinBaseTool):
    name: str = "anakin_ai_visibility_search"
    description: str = (
        "Ask several AI answer engines (such as ChatGPT, Gemini and Google AI Overview) "
        "the same question and compare their answers, with a synthesis of where they "
        "agree. For brand and AI-SEO visibility checks. Takes 1-2 minutes; billed per "
        "engine."
    )
    args_schema: type[BaseModel] = AiVisibilitySearchInput

    def _call(self, args: AiVisibilitySearchInput) -> Any:
        result = _jsonable(
            self.client.ai_visibility.search(args.query, sources=args.sources, country=args.country)
        )
        if not args.include_full_content:
            for engine in result.get("results", []):
                engine.pop("full_content", None)
        return result


class NoInput(BaseModel):
    pass


class AnakinAiVisibilitySourcesTool(AnakinBaseTool):
    name: str = "anakin_ai_visibility_sources"
    description: str = (
        "List the AI answer engines anakin_ai_visibility_search can ask, with their slugs."
    )
    args_schema: type[BaseModel] = NoInput

    def _call(self, args: NoInput) -> Any:
        return self.client.ai_visibility.sources()


# ── Saved browser sessions and browser tasks ──────────────────────────────


class SessionListInput(BaseModel):
    domain: str | None = Field(None, description='Only sessions for this site, e.g. "amazon.com".')


class AnakinSessionListTool(AnakinBaseTool):
    name: str = "anakin_session_list"
    description: str = (
        "List the user's saved browser sessions (encrypted login states created in the "
        "Anakin dashboard). Pass a session's id as session_id to anakin_scrape, "
        "anakin_crawl, anakin_monitor_create or anakin_browser_task to reach pages "
        "behind that login."
    )
    args_schema: type[BaseModel] = SessionListInput

    def _call(self, args: SessionListInput) -> Any:
        return self.client.sessions.list(domain=args.domain)


class SessionDeleteInput(BaseModel):
    session_id: str = Field(description="The session to delete (from anakin_session_list).")


class AnakinSessionDeleteTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_session_delete"
    description: str = (
        "Permanently delete a saved browser session. The user must log in again in the "
        "dashboard to recreate it, and monitors using it lose access. Confirm with the "
        "user first."
    )
    args_schema: type[BaseModel] = SessionDeleteInput

    def _call(self, args: SessionDeleteInput) -> Any:
        self.client.sessions.delete(args.session_id)
        return {"id": args.session_id, "deleted": True}


class BrowserTaskInput(BaseModel):
    prompt: str = Field(description="The task, specific about the goal and what to return.")
    url: str | None = Field(None, description="Where to start.")
    session_id: str | None = Field(None, description="Saved browser session for logged-in tasks.")
    output_schema: dict[str, Any] | None = Field(
        None, description="JSON Schema for a structured result."
    )
    max_steps: int | None = Field(None, ge=1, description="Cap on browser steps.")
    timeout_ms: int | None = Field(None, ge=1000, description="Task timeout in milliseconds.")


class AnakinBrowserTaskTool(AnakinBaseTool):
    writes: ClassVar[bool] = True
    name: str = "anakin_browser_task"
    description: str = (
        "Have an AI agent drive a real cloud browser for a multi-step task (navigate, "
        "click, type, extract). Runs up to about 5 minutes. Use only when anakin_scrape "
        "can't do it and no Wire action exists. Never put passwords in the prompt: use "
        "session_id. Payments are refused."
    )
    args_schema: type[BaseModel] = BrowserTaskInput

    def _call(self, args: BrowserTaskInput) -> Any:
        # The SDK has no browser-task method, so this calls the REST API directly.
        self._refuse_if_financial("task", args.prompt, args.url)
        api_key = self.api_key or os.environ.get("ANAKIN_API_KEY")
        if not api_key:
            raise AnakinError(
                "No Anakin API key: pass api_key or set ANAKIN_API_KEY. "
                "Get one free at https://anakin.io/dashboard."
            )
        body = {"prompt": args.prompt, "async": True}
        body.update(args.model_dump(exclude={"prompt"}, exclude_none=True))

        with httpx.Client(
            base_url=self.base_url.rstrip("/"), headers={"X-API-Key": api_key}, timeout=30.0
        ) as http:
            accepted = _request(http, "POST", "/ai/evaluate", json=body)
            workflow_id = accepted.get("workflow_id")
            if not workflow_id:
                return accepted
            deadline = time.monotonic() + BROWSER_TASK_MAX_SECONDS
            while time.monotonic() < deadline:
                job = _request(http, "GET", f"/ai/jobs/{quote(str(workflow_id), safe='')}")
                status = job.get("status")
                if status == "completed":
                    return job.get("result", job)
                if status in ("failed", "timed_out"):
                    raise AnakinError(
                        f"Browser task {status}: {job.get('error') or 'unknown error'}",
                        code=str(status).upper(),
                    )
                time.sleep(BROWSER_TASK_POLL_SECONDS)
        raise AnakinError("Browser task still running after 6 minutes.", code="TIMEOUT")


def _request(http: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        res = http.request(method, path, **kwargs)
    except httpx.HTTPError as err:
        raise AnakinError(f"Could not reach the Anakin API: {err}") from err
    try:
        data = res.json() if res.content else {}
    except ValueError:
        data = {}
    if res.status_code >= 400:
        detail = data.get("message") or data.get("error") if isinstance(data, dict) else None
        if isinstance(detail, dict):
            detail = detail.get("message")
        raise AnakinError(
            detail or f"Anakin API returned {res.status_code}",
            status_code=res.status_code,
            body=data,
        )
    return data if isinstance(data, dict) else {"result": data}


# ── All tools ─────────────────────────────────────────────────────────────

ALL_TOOLS: tuple[type[AnakinBaseTool], ...] = (
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinMapTool,
    AnakinCrawlTool,
    AnakinAgenticSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireCatalogTool,
    AnakinWireReadActionTool,
    AnakinWireWriteActionTool,
    AnakinWireIdentitiesTool,
    AnakinWireLoginTool,
    AnakinWireBuildTool,
    AnakinWireBuildStatusTool,
    AnakinMonitorCreateTool,
    AnakinMonitorListTool,
    AnakinMonitorChangesTool,
    AnakinMonitorControlTool,
    AnakinAiVisibilitySearchTool,
    AnakinAiVisibilitySourcesTool,
    AnakinSessionListTool,
    AnakinSessionDeleteTool,
    AnakinBrowserTaskTool,
)


def get_anakin_tools(
    api_key: str | None = None,
    *,
    read_only: bool = False,
    allow_financial_actions: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    poll_timeout: float = 600.0,
    client: Anakin | None = None,
) -> list[AnakinBaseTool]:
    """
    Every Anakin tool, sharing one API client.

    Args:
        api_key: Anakin API key. Defaults to the ANAKIN_API_KEY environment variable.
        read_only: Leave out the tools that change something on a website or in the
            user's Anakin account, or start a long paid job.
        allow_financial_actions: Allow Wire writes, Wire builds and browser tasks
            that look like a payment or money transfer. Off by default.
        base_url: Anakin API base URL.
        poll_timeout: Seconds to wait for async jobs (scrape, crawl, research).
        client: An existing `anakin.Anakin` client to use instead.
    """
    shared = client or Anakin(api_key=api_key, base_url=base_url, poll_timeout=poll_timeout)
    return [
        cls(
            client=shared,
            api_key=api_key,
            base_url=base_url,
            poll_timeout=poll_timeout,
            allow_financial_actions=allow_financial_actions,
        )
        for cls in ALL_TOOLS
        if not (read_only and cls.writes)
    ]
