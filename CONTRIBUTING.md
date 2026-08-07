# Contributing

Thanks for your interest in improving Stock AI Predictor.

## Development Workflow

1. Fork and create a feature branch.
2. Keep changes focused and atomic.
3. Add or update docs/tests for any behavioral changes.
4. Run local checks before opening a PR.

## Local Validation

```bash
python -m compileall src scripts examples
PYTHONPATH=src python -m unittest discover -s tests
```

## Coding Standards

- Use Python type hints for public interfaces.
- Add docstrings to modules, classes, and non-trivial functions.
- Add inline comments only where RL or data logic is complex.
- Prefer explicit error messages with actionable context.

## Pull Request Checklist

- [ ] Scope is focused
- [ ] Documentation updated
- [ ] Tests added/updated when needed
- [ ] No secrets committed (`.env` stays local)
