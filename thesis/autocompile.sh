#!/bin/bash
# Run this to automatically recompile thesis each time it is updated

file="prace"
bib="bibliography"

if [ -f $1.tex ]; then
    echo "Compiling $file.tex automatically based on updates of $1.tex"
    file_check=$1
else
    file_check=$file
fi

lastUpdate=0
lastBibUpdate=0
while [ true ]; do
    sleep 1
    nowUpdate="`stat --format='%Y' "$file_check.tex"`"
    if [ $lastUpdate != $nowUpdate ]; then
        # recompile 
        lastUpdate=$nowUpdate
        pdflatex "$file.tex" 1> /dev/null
        echo "Updated tex: $nowUpdate"
    fi
    bibUpdate="`stat --format='%Y' "$bib.bib"`"
    if [ $lastBibUpdate != $bibUpdate ]; then
        lastBibUpdate=$bibUpdate
        biber "$file.bcf"
        pdflatex "$file.tex" 1> /dev/null
        pdflatex "$file.tex" 1> /dev/null
        echo "Updated biber: $bibUpdate"
    fi
done
