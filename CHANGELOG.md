# Changelog

## 0.1.0 (unreleased)

- All 22 Anakin tools as CrewAI tools, with the hosted MCP server's names
  prefixed `anakin_`, built on anakin-sdk 0.2.
- `get_anakin_tools()` returns them all on one shared client; `read_only=True`
  leaves out the 7 that change something.
- Wire writes, Wire builds and browser tasks refuse payments and money
  transfers unless `allow_financial_actions=True`.
