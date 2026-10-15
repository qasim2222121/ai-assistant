# ai-assistant
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score (0-100)**, a category
breakdown, matched/missing keywords and **concrete rewrite suggestions**.
Built with [Streamlit](https://streamlit.io) and Google **Gemini Flash**.

## Features
- PDF / DOCX / TXT upload
- Optional job description for an accurate keyword match
- Score breakdown: keywords, content impact, formatting, structure
- Before/after rewrite suggestions
- Rule-based checks (email, phone, sections, bullets, metrics) that need no AI

## How the score works
Gemini rates four categories from 0-100. The overall score is calculated in code
as a weighted average: keywords 30%, content 30%, formatting 20%, structure 20%.
This is an *estimate* of ATS-friendliness, not the output of a real ATS.

## Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

**Option A, environment variable**
```bash
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

**Option B, secrets file** (create `.streamlit/secrets.toml`, never commit it)
```toml
GEMINI_API_KEY = "your-key"
```

**Option C**, paste the key into the app sidebar.

Run:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (the repo must contain `app.py` and `requirements.txt`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings -> Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration
- The model defaults to `gemini-2.5-flash` and can be changed in the sidebar.
  If Google retires a model name, enter the current Flash model name there, or
  change `DEFAULT_MODEL` in `app.py`.

## Limitations
- Scanned/image-only PDFs have no extractable text (an ATS can't read them either).
- Resume text is sent to the Gemini API; don't upload data you aren't comfortable sharing.
- AI feedback can be imperfect. Review suggestions before using them.
