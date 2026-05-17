# Contributing

Thanks for taking a look at NeuroForge.  The project values small, well-tested patches over broad rewrites.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
ruff check .
```

## Pull Requests

- Keep changes scoped to one concern.
- Add or update tests for behavior changes.
- Include benchmark notes for performance-sensitive code.
- Prefer plain implementation notes over clever comments.
- Call out known limitations directly; TODOs are welcome when they mark real follow-up work.

## Style

Typed Python is preferred, but readability wins over type gymnastics.  Public APIs should include docstrings when behavior is not obvious from the signature.
