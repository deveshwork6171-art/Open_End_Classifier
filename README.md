# AI Open-Ended Response Validator (Streamlit)

This app adapts the uploaded Colab notebook into a Streamlit interface. The notebook uses Gemini (`gemini-2.5-flash`) and classifies open-ended survey responses using a question-specific prompt, reference examples, numbered batches, and `VALID`, `INVALID`, `GARBAGE`, and `NON_INSIGHTFUL` categories. This app retains those core principles and adds `NEEDS_REVIEW` for ambiguous responses, malformed model output, or failed batches.

## Features

- Upload `.xlsx`, `.xls`, or `.csv` response files.
- Choose the column containing the open-ended responses.
- Enter the exact survey question.
- Add reference examples in a text box and/or upload a reference file.
- Enter a Gemini API key into a password field.
- Choose the Gemini model, batch size, and retry count.
- Classify unique responses once and reuse the result for repeated answers.
- Preserve all original rows and columns.
- Add `Classification`, `Classification_Reason`, and `Validation_Question`.
- View category counts and filter the result table.
- Download complete and flagged reports in Excel or CSV.

## 1. Install Python

Use Python 3.10 or newer. Create a folder for the app and copy `app.py` and `requirements.txt` into it.

## 2. Create and activate a virtual environment (recommended)

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Start Streamlit

Run this command from the folder containing `app.py`:

```bash
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the local URL printed in the terminal.

## 5. Use the app

1. Enter the exact survey question.
2. Upload your response file.
3. Select the column that contains the open-ended responses.
4. Paste valid examples (one per line or comma-separated), or upload a reference file and select its reference column.
5. Enter your Gemini API key in the sidebar.
6. Select the model and batch size.
7. Click **Validate Responses**.
8. Review the results and download the complete or flagged report.

## Classification categories

- `VALID`: A meaningful answer that fits the survey question.
- `INVALID`: Understandable but irrelevant or not a suitable answer to the question.
- `GARBAGE`: Random, nonsensical, unreadable, or corrupted text.
- `NON_INSIGHTFUL`: A blank, explicit non-answer, or uncertainty/refusal response.
- `NEEDS_REVIEW`: Ambiguous response, unparseable model output, or batch/API error.

The reference examples are treated as guidance rather than a complete whitelist. A legitimate response not listed in the references can still be classified as valid.

## API key safety

- Do not hardcode API keys into `app.py`.
- The API key field uses a password input and the key is not included in downloaded files.
- The app does not intentionally print or log the API key. It is held in memory for the current Streamlit session.
- For shared or hosted deployments, configure provider secrets using the hosting platform's secret manager rather than asking every user to enter a key.
- The notebook you uploaded appears to contain an API-key-like value embedded in an input prompt. If that value is a real key, revoke/rotate it in the provider console and avoid reusing it.

## Notes and limitations

- The model makes probabilistic classifications. Review flagged and borderline responses before removing respondents.
- Batch size affects prompt size, API usage, speed, and reliability. Start with 20–50 responses per batch for typical open ends.
- Very long individual responses or very large batches may exceed model limits; reduce the batch size if that happens.
- Excel results contain the original data plus classification fields and a summary worksheet.
- The app has not been tested with your live API key; successful classification requires a valid Gemini API key, model access, and network connectivity.
