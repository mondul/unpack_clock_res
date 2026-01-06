param([string]$Path = ".venv")
python -m venv $Path
Write-Host "Virtual environment created at $Path"
Write-Host "To activate in PowerShell: .\$Path\Scripts\Activate.ps1"
Write-Host "Then install dependencies: pip install -r requirements.txt"
