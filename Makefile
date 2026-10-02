.PHONY: install build build-openfda test eval eval-gemini api dashboard up
install:        ; pip install -r requirements.txt
build:          ; python -m scripts.build_all --source synthetic
build-openfda:  ; python -m scripts.build_all --source openfda --per-drug 3000
test:           ; python -m pytest -q
eval:           ; python -m evals.run_eval --backend offline
eval-gemini:    ; python -m evals.run_eval --backend gemini
api:            ; uvicorn pharmawatch.api:app --reload --port 8000
dashboard:      ; streamlit run dashboard/app.py
up:             ; docker compose up --build
