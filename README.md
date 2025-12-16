Quick setup and run instructions (PowerShell)

1) Activate your virtualenv (if you have one):

```powershell
# from project root
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
# If you want LanguageTool suggestions, ensure Java (JRE) is installed and then:
# python -m pip install language-tool-python
```

3) Run the script:

```powershell
python Final_Writing_Feedback.py
```

Notes:
- The script will prefer spaCy-based analysis when the `en_core_web_sm` model is installed; otherwise it safely falls back to regex-based checks and prints a note.
- LanguageTool (language-tool-python) provides stronger grammar/style suggestions but requires a Java runtime (JRE) installed on your machine; if Java isn't present, the code will skip LanguageTool suggestions.
- seaborn is optional for nicer plots; if it's missing the script uses matplotlib styles.
