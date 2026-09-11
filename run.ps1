# run.ps1 — Windows helper: sets UTF-8 console encoding before running scripts
# Usage: .\run.ps1 scripts\02_data_pipeline.py --brand AmazonHelp --sample-size 50000
#        .\run.ps1 scripts\06_inference.py --message "my order hasn't arrived"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

C:\Users\Maxpr\anaconda3\python.exe @args
