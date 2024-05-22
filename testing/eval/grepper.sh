#!/bin/bash

grep -v -e "clingo version" -e "Reading from" -e "Solving..." -e "SATISFIABLE" -e "Calls" -e "CPU Time" -- $1
