pakdataset.csv
      ↓
Cleaning
      ↓
Duplicate removal
      ↓
Target validation
      ↓
Leakage check
      ↓
Train / Validation / Test
      ↓
 ┌──────────────┬──────────────┬────────────────────┐
 │ Random Forest│     SVM      │ New Tabular        │
 │              │              │ Transformer        │
 └──────────────┴──────────────┴────────────────────┘
                      ↓
              Same test dataset
                      ↓
             Accuracy / Precision
             Recall / F1 / ROC-AUC
                      ↓
                Best Model
                      ↓
             Save for Sneha

## Sneha screening flow

The web page offers both the supportive conversation and a structured
questionnaire. The questionnaire uses the same 25 required features, answer
choices, validation, saved model, and screening-result explanation as the
conversation. Patients must sign in to submit a screening; submitted answers
and server-generated depression results are saved to the patient's account.
A nonzero answer to the self-harm question interrupts screening without
producing a model result and displays an immediate-safety message.
In conversation mode, Sneha also has a small animated portrait that moves
while preparing and responding; it uses inline artwork and CSS animation,
with no external image service. Conversation replies use Azure Speech neural
text-to-speech with the `en-IN-NeerjaNeural` voice when configured. A voice
toggle can pause or resume spoken replies. If Azure Speech is unavailable,
the browser's built-in speech is used as a fallback.

To enable Azure Speech, create a Speech resource on its F0 free tier, then set
the resource key and region in the PowerShell window used to start the app:

```powershell
$env:AZURE_SPEECH_KEY = "<your speech resource key>"
$env:AZURE_SPEECH_REGION = "<your speech resource region>"
python -m uvicorn app:app --reload
```

The key stays on the server and must not be committed or put in frontend code.
Azure applies the quota for the selected resource tier; check the current
[Azure Speech pricing and quotas](https://azure.microsoft.com/en-us/pricing/details/speech/)
before use. The app limits individual text requests to 4,000 characters but
does not track monthly usage across app restarts. If credentials are missing,
the app reports that Azure is unavailable and falls back to browser speech.

The conversation engine collects and validates the original 25 questionnaire
answers, then runs the saved Tabular Transformer Experiment A checkpoint and
its fitted preprocessing. The separate `FTTransformer` class does not have a
corresponding saved checkpoint in this project, so it is not used by the chat.
Question text comes from the reviewed templates instead of an LLM rewrite,
so the app avoids an extra generation request at the start and after each
accepted answer. Explicit gestational-age durations in days are converted to
weeks (days divided by seven); a bare number is not reinterpreted as days.
The screening result is an estimate, not a diagnosis. A self-harm disclosure
interrupts and stops screening for that session; no model result is produced.
If extraction finds one strongly supported but uncertain answer, Sneha asks the
patient to confirm it before storing it. Unsupported or low-confidence
suggestions are not used, and answers are never selected randomly.
After three unclear answers to the same feature, finite-choice questions show
selectable options. Numeric questions instead give an example format and still
require the patient to enter their own value. Choosing a nonzero frequency for
the self-harm item stops screening and starts the safety follow-up.

## Evaluation limitations and next steps

The repository's `data/pakdataset.csv` has 14,008 rows, of which 7,000 are
exact duplicate rows. `Scalling` equals the sum of the nine symptom answers on
every row, and `Labelling` matches a cutoff at 10 on that sum on every row:
scores below 10 are `Not`; scores of 10 or more are `Depressed`. This supports
that the saved Experiment A model reproduces this dataset's label rule; it
does not establish independent clinical prediction. The frontend discloses
this limitation.

The repository does not include a source citation or patient/participant ID
column for this dataset, so its provenance and patient-level split integrity
cannot be independently established from these files. `train_model.py` now
deduplicates identical model-input/label rows and keeps identical input
profiles in the same train/validation/test partition. This is not a
patient-grouped split and does not replace validation on an independently
collected patient cohort. No model was retrained or overwritten.

Before making clinical-performance claims or using this with patients:

1. Obtain the dataset citation and documentation from its provider, and
   independently confirm how `Labelling` was assigned.
2. Have a qualified clinician define and approve the intended screening
   questionnaire, scoring, thresholds, units, and safety/referral workflow.
3. Evaluate on a properly governed, independently labeled external cohort
   with patient-separated splits and report sensitivity, specificity,
   calibration, and uncertainty. This project does not contain that cohort.
4. Test answer extraction on a clinician-reviewed synthetic/consented gold
   set. Current automated tests cover code behavior, not Qwen's empirical
   extraction accuracy.
5. Compare local and hosted LLMs only on that same approved test set. No
   hosted provider is configured here, and real patient answers should not be
   sent to an external service without privacy review and informed consent.

Run the lifecycle and saved-model integration tests from this folder:

```powershell
python -m unittest discover -s src -p "test*.py" -v
```

Configure the database and application secrets before first startup. Copy
`.env.example` to `.env`, replace every placeholder with your own values, and
create the configured MySQL database. Then apply migrations and start the app:

```powershell
Copy-Item .env.example .env
python -m alembic upgrade head
python -m uvicorn app:app --reload
```

The root URL `http://127.0.0.1:8000/` opens the separate login page. Patients
must sign in before opening `/screening`; successful patient login opens their
dashboard, where they can choose **Start screening**. Conversation sessions are
kept in memory and expire after 30 minutes of inactivity.

Patient flow:

- Register or log in as a patient.
- After authentication, the app redirects the patient to their account portal so
  screening and saved history remain tied to that user record.
- Every screening result is saved with the response details and a clinician-
  friendly depression prediction summary.
- Admin users can review all patient answers and saved outcomes from the admin
  dashboard, including the charted depression distribution and recent activity.
- The admin dashboard includes a clear-data action to remove saved screening
  data when a fresh review cycle is needed.

Doctor authentication:

- The registration page supports both patient and doctor accounts. Doctor
  registrations collect professional credentials and remain pending; they do
  not grant access to patient records automatically.
- An administrator can review submitted doctor credentials in the
  **Doctor verification** section of `/admin`. Verifying activates the account;
  rejecting it blocks sign-in.
- Apply schema updates with `python -m alembic upgrade head` after setting
  `DATABASE_URL` in `.env`. Existing databases are not altered by
  `create_all()` alone.
- Patients can explicitly share history with a verified clinician from the
  patient portal and can revoke access. Clinicians can only search or open
  records while that consent is active. Sensitive record views and clinical
  notes/recommendations are audited.
- Patients can update their profile details and review their saved answers,
  screening results, clinician notes, and recommendations. Clinicians can add
  assessment-linked notes and recommendations without changing the model output.
- Set `COOKIE_SECURE=true` when serving the application over HTTPS. Production
  startup requires a non-placeholder `SECRET_KEY`.
- Optional Google sign-in: create a Google OAuth client, set
  `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REDIRECT_URI` in
  `.env`, and add the exact callback URI to the Google OAuth client's
  authorized redirect URIs (for local development,
  `http://127.0.0.1:8000/auth/google/callback`). Restart FastAPI after changing
  `.env`; it is loaded from the project directory regardless of the launch
  directory. The Google button appears only when both client credentials are
  configured. Google sign-in requires a verified email already registered
  with Sneha; it does not create accounts or bypass doctor approval and patient
  consent. Until credentials are configured, the login page explains that
  Google sign-in is unavailable and password login continues to work.

The current repository contains an antenatal depression predictor only. It
does not include a validated anxiety questionnaire or anxiety prediction
artifact, so this application does not report an anxiety prediction. Screening
outputs are estimates and are not diagnoses.