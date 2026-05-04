$env:PATH += ";C:\Users\rvabh\AppData\Roaming\TinyTeX\bin\windows"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
Write-Host "Done. Check main.pdf"
