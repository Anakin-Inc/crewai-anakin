# crewai-anakin

CrewAI tools for [Anakin](https://anakin.io): scrape, search, map and crawl the
web, run deep research, use pre-built Wire actions on hundreds of sites, monitor
pages for changes, compare AI answer engines and drive a cloud browser.

All 22 tools of Anakin's hosted MCP server (`mcp.anakin.io`), as CrewAI tools,
built on the official [`anakin-sdk`](https://pypi.org/project/anakin-sdk/) client.

## Install

```bash
pip install crewai-anakin
export ANAKIN_API_KEY=...   # free at https://anakin.io/dashboard
```

Requires Python 3.10 to 3.13 (the versions CrewAI supports) and `crewai` 1.0 or later.

## Use with an agent

```python
from crewai import Agent, Crew, Task
from crewai_anakin import get_anakin_tools

researcher = Agent(
    role="Market researcher",
    goal="Answer questions with facts from the web, citing the pages used",
    backstory="An analyst who checks sources before answering.",
    tools=get_anakin_tools(),
)

task = Task(
    description="What do the top three CRM vendors charge per seat?",
    expected_output="A table of vendor, plan and price per seat, with source URLs.",
    agent=researcher,
)

print(Crew(agents=[researcher], tasks=[task]).kickoff())
```

`get_anakin_tools()` returns every tool, sharing one API client. To give an
agent only some, use the tool classes directly:

```python
from crewai_anakin import AnakinCrawlTool, AnakinScrapeTool, AnakinSearchTool

tools = [AnakinSearchTool(), AnakinScrapeTool(), AnakinCrawlTool()]
```

## Options

```python
get_anakin_tools(
    api_key=None,  # defaults to ANAKIN_API_KEY
    read_only=False,  # True leaves out the 7 tools that change something
    allow_financial_actions=False,  # see "Payments" below
    poll_timeout=600.0,  # seconds to wait for scrapes, crawls and research
    client=None,  # or pass your own anakin.Anakin client
)
```

The same settings (`api_key`, `poll_timeout`, `allow_financial_actions`,
`base_url`, `client`) work on each tool class.

**`read_only=True`** leaves out the tools that change something on a website or
in the user's Anakin account, or that start a long paid job: `anakin_wire_write_action`,
`anakin_wire_login`, `anakin_wire_build`, `anakin_monitor_create`,
`anakin_monitor_control`, `anakin_session_delete` and `anakin_browser_task`.

**Payments.** Like Anakin's hosted MCP server, `anakin_wire_write_action`,
`anakin_wire_build` and `anakin_browser_task` refuse requests that look like
they complete a payment or move money (checkout, place order, pay now, wire
transfer, ...). "Add to cart" and every read are allowed.
`allow_financial_actions=True` turns this off.

## Tools

Each tool returns JSON. Tool names match the hosted MCP server's, prefixed with
`anakin_`.

| Tool | Class | What it does |
|---|---|---|
| `anakin_scrape` | `AnakinScrapeTool` | One page as clean markdown, optionally with AI-extracted JSON |
| `anakin_search` | `AnakinSearchTool` | Web search: URLs, titles and snippets, returned immediately |
| `anakin_map` | `AnakinMapTool` | Every URL on a site, to decide what to crawl |
| `anakin_crawl` | `AnakinCrawlTool` | Many pages of one site as markdown |
| `anakin_agentic_search` | `AnakinAgenticSearchTool` | Deep research across many sources (1-5 minutes) |
| `anakin_wire_discover` | `AnakinWireDiscoverTool` | Find pre-built Wire actions for a task |
| `anakin_wire_catalog` | `AnakinWireCatalogTool` | Every Wire site, or one site's actions and parameters |
| `anakin_wire_read_action` | `AnakinWireReadActionTool` | Run a Wire action that reads data |
| `anakin_wire_write_action` | `AnakinWireWriteActionTool` | Run a Wire action that changes something on the site |
| `anakin_wire_identities` | `AnakinWireIdentitiesTool` | Saved site logins and their credential IDs |
| `anakin_wire_login` | `AnakinWireLoginTool` | Sign in to a site and get a credential ID |
| `anakin_wire_build` | `AnakinWireBuildTool` | Ask Wire to build actions for a new site |
| `anakin_wire_build_status` | `AnakinWireBuildStatusTool` | Progress of Wire builds |
| `anakin_monitor_create` | `AnakinMonitorCreateTool` | Watch a page, site or Wire action on a schedule |
| `anakin_monitor_list` | `AnakinMonitorListTool` | Your monitors, or one monitor's settings |
| `anakin_monitor_changes` | `AnakinMonitorChangesTool` | Changes a monitor detected |
| `anakin_monitor_control` | `AnakinMonitorControlTool` | Pause, resume, run now or delete a monitor |
| `anakin_ai_visibility_search` | `AnakinAiVisibilitySearchTool` | Ask ChatGPT, Gemini and Google AI Overview the same question |
| `anakin_ai_visibility_sources` | `AnakinAiVisibilitySourcesTool` | The answer engines available |
| `anakin_session_list` | `AnakinSessionListTool` | Saved browser sessions, for pages behind a login |
| `anakin_session_delete` | `AnakinSessionDeleteTool` | Delete a saved browser session |
| `anakin_browser_task` | `AnakinBrowserTaskTool` | An AI agent that drives a real browser through a multi-step task |

## Errors

Anakin API errors raise `anakin.errors.AnakinError` (and its subclasses, such as
`AuthenticationError` or `InsufficientCreditsError`). CrewAI passes a tool's
error back to the agent, which can retry or explain it.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Releases publish to PyPI from GitHub Actions when a `v*` tag matching the
version in `pyproject.toml` is pushed.

## License

MIT
