#!/bin/bash
# Run this to automatically recompile thesis each time it is updated

file="prace.tex"

lastUpdate=0
while [ true ]; do
    sleep 1
    nowUpdate="`stat --format='%Y' $file`"
    if [ $lastUpdate != $nowUpdate ]; then
        lastUpdate=$nowUpdate
        pdflatex $file 1> /dev/null
        echo "Updated: $nowUpdate"
        echo "-----------------------------------------"
    fi
done
