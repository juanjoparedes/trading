# Trading Agent — Project Instructions

## Security

- Never implement live trading during the early phases.
- Never execute trades with real money without explicit authorization.
- Never store API keys, secrets, tokens, or passwords in Git.
- Never create real credentials.
- Never print credentials in logs.
- Use `.env` for secrets when they are needed in the future.
- Keep `.env.example` free of secret values.

## Quantitative Research

- Avoid look-ahead bias.
- Avoid survivorship bias when applicable.
- Avoid overfitting.
- Separate training and evaluation data when applicable.
- Every strategy must be backtestable.
- Every strategy must have tests.
- Results must be reproducible.
- Financial hypotheses must be documented.

## Architecture

Keep these concerns separate:

- market data
- indicators
- strategies
- signals
- risk
- backtesting
- execution
- portfolio
- journal

A strategy must never execute an order directly. Risk management must remain separate from strategy logic.

## Development

- Use Python.
- Keep code modular.
- Use type hints where appropriate.
- Write tests for critical components.
- Keep functions small and understandable.
- Do not introduce unnecessary dependencies.
- Run tests after every relevant change.
- Do not modify functionality outside the requested task scope.

## Project Control

The project is developed through milestones. Do not automatically advance to the next milestone. When a task is complete, stop and report the result.
