# Resume Screening Agent

Professional AI-assisted resume screening web app (Streamlit) for ranking candidates against a job description.

This project provides a rapid, visual interface to upload multiple resumes (PDF/DOCX), extract and match skills using spaCy + skillNer, compute semantic similarity via sentence-transformer embeddings, and produce an AI-powered candidate analysis using an LLM wrapper.

---

## Key features

- Batch upload of resumes (PDF/DOCX).
- Skill extraction using spaCy + skillNer with fallback handling.
- Semantic similarity scoring using SentenceTransformers (`all-MiniLM-L6-v2`).
- AI analysis of resume vs JD (LLM), with robust JSON parsing and retry logic.
- Interactive dashboard with charts, candidate cards, and shortlist management.
- Light/Dark theme toggle persisted in session state.
- Export candidate analysis as JSON / CSV downloads.

---

## File structure

- `app.py` — Main Streamlit application (upload, screening, results, shortlist UI).
- `ui_components.py` — Reusable UI helpers (CSS injection, Plotly charts, candidate card).
- `analytics.py` — Small heuristics helpers (resume quality, skill counting, overall match).
- `requirements.txt` — Python dependencies.
- `skill_db_relax_20.json`, `token_dist.json` — SkillNer related DB / token data (if present).

---

## Quick setup (Windows / PowerShell)

Recommended: create and activate a virtual environment first.

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model (required by the app)
python -m spacy download en_core_web_sm
```

Notes:
- Some packages (e.g., `faiss-cpu`, `sentence-transformers`) may have platform-specific wheel availability. If installation fails, consult their project pages for platform-specific instructions.

---

## Run the app

From the project root (PowerShell):

```powershell
# Run normally
streamlit run app.py

# Or run with debug logging and save logs (useful for diagnosing startup failures)
streamlit run app.py --logger.level=debug > streamlit_debug.log 2>&1

# To view last 200 lines of the log (PowerShell)
Get-Content .\streamlit_debug.log -Tail 200
```

If Streamlit exits with code 1 or shows unexpected behavior, run the debug command above and attach the last ~200 lines of `streamlit_debug.log` when asking for help.

---

## Common troubleshooting

1. SkillNer / spaCy initialization errors
   - Symptom: Streamlit prints lines like "loading full_matcher ..." and then exits with code 1.
   - Steps:
     - Ensure `en_core_web_sm` is installed (see Quick setup above).
     - Check `streamlit_debug.log` for the Python traceback. The traceback will indicate if SkillExtractor failed due to missing files, incompatible versions, or other runtime exceptions.
     - The app wraps `SkillExtractor` initialization in a try/except and stores the error message; during runtime the app will show a UI warning if skill extraction is unavailable. Use logs to diagnose further.

2. LLM JSON parsing issues
   - The app sends a strict prompt asking for JSON, but models can return prose. The code includes retry logic which asks the LLM to convert its raw output to valid JSON once.
   - If analysis still fails, the UI stores a fallback JSON including embedded `llm_raw` for debugging.

3. Dependency install failures
   - If `pip install -r requirements.txt` fails for binary packages (faiss, specific torch versions, etc.), review the failing package's install docs and consider installing a CPU-only or platform-specific wheel.

---

## Development notes

- Theme: `st.session_state.theme` holds the user's choice (`'dark'` or `'light'`). CSS is injected from `ui_components.inject_css(theme)`.
- Skill extraction: `app.py` calls `SkillExtractor` from `skillNer`. If your environment doesn't require skillNer, the app will continue in degraded mode (skill list empty) but retain LLM analysis and similarity scoring.
- LLM: The app uses `ChatOllama` from `langchain_community` as the default wrapper in this repo; you may replace it with your preferred LLM wrapper as long as it returns text or an object with a `.content` attribute.

---

## How to contribute

1. Fork the repo and create a feature branch.
2. Run and test locally using the steps above.
3. Open a PR with a clear description of the change.

---

## License & Contact

This project is provided as-is for internal use. If you have questions, paste your `streamlit_debug.log` output when reporting runtime failures and include your Python version and OS details.

