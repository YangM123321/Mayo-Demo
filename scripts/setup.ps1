# Creates .venv and installs Python deps
Continue = "Stop"

if (-Not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "Python not found. Install Python 3.11+ and ensure 'python' is on PATH."
}

python -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r .\requirements.txt

# Optional NLTK downloads (safe to skip)
try {
  python - << 'PY'
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK resources installed.")
PY
} catch { Write-Host "Skipping NLTK downloads." }

Write-Host "Setup complete."
