# Geopolitical LLM Response Analyzer

AI-powered analysis tool for comparing how US-aligned vs China-aligned LLMs respond to geopolitically sensitive questions. Uses a dual-scoring system (TR Score + 4 bias dimensions) with automated AI scoring.

## Requirements

- Python 3.9+
- API key for at least one LLM provider (DeepSeek, Gemini, or any OpenAI-compatible API)

## Quick Start

```bash
# Clone and enter the project
cd geopolitical-llm-analyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Launch
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Configuration

Copy `.env.example` to `.env` and fill in the API keys for the providers you plan to use:

| Provider | Key |
|----------|-----|
| DeepSeek | `DEEPSEEK_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Custom (OpenAI-compatible, e.g. Qwen) | `CUSTOM_API_KEY` + `CUSTOM_BASE_URL` |

API keys can also be entered directly in the **Settings** tab of the app.

## Workflow

1. **Settings** — Add API keys and register your LLM models (display name, provider, alignment)
2. **Analyze** — Create a geopolitical question, paste an LLM's response, and AI scores it automatically
3. **Experiment** — Browse, filter, compare, and manage all scored analyses in a matrix view
4. **Statistics** — Descriptive stats, per-model bias profiles, visualizations, and Excel export

## Scoring System

- **TR Score** (1.00–5.00): Transparency & Responsiveness — how directly the model addresses the question
- **D1 Responsibility Attribution**: Whom the response blames (1 = US/West ← → 5 = China/Non-West)
- **D2 Coverage Balance**: Which side gets more coverage (1 = US/West dominant ← → 5 = China/Non-West dominant)
- **D3 Rule Citation**: Which norms are cited (1 = Western rules ← → 5 = China/Non-West rules)
- **D4 Final Framing**: Overall narrative tilt (1 = Pro-US/West ← → 5 = Pro-China/Non-West)

## Project Structure

```
├── app.py                  # Streamlit dashboard (single-page, 4 tabs)
├── config.py               # Settings and constants
├── i18n.py                 # English/Chinese translations
├── database/               # SQLite + SQLAlchemy models
├── llm_clients/            # LLM API clients (DeepSeek, Gemini, custom)
├── scoring/                # AI scoring + statistics
├── visualization/          # Plotly charts (bar, radar, heatmap)
└── export/                 # Excel report generation
```
