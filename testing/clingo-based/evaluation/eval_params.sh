#!/bin/bash

folder="noFolder"
params="evaluation/params.txt"

countdown() (
	trap - INT
	secs=5
	while [ $secs -gt 0 ]; do
		echo -ne "$secs\033[0K\r"
		sleep 1
		: $((secs--))
	done
)

i=0
total="`wc -l $params`"
skip="True"

cat $params | while read line; do
	i="$[ $i + 1 ]"
	# Line specifies new test
	if [ "${line:0:1}" == "!" ]; then
		folder="${line:2}"
		mkdir -p $folder

		if [ "$?" == 0 ]; then
			echo "Starting <$folder>..."
			skip="False"
		else
			echo "Remove <$folder> before running"
			skip="True"
		fi
		continue
	fi

	if [ "$skip" == "True" ]; then
		continue
	fi

	# Line specifies parameters
	echo "Line $i/$total"
	echo "Parameters : $line" >> $folder/output.txt
	./evaluator.py $line >> $folder/output.txt

	error=$?
	break="True"
	case "$error" in
		1|11|75|65)
			break="False"
			echo "Interruption on $line of $folder. Countdown before next parameter."
			trap '' INT
			countdown || { break="True"; echo "Countdown interrupted, skipping remaining"; }
			trap - INT
			;;
		20|30)
			# normal run Unsat | Sat
			break="False"
			;;
		*)
			break="False"
			echo "Signal: $error"
			;;
	esac
	if [ "$break" == "True" ]; then
		mv $folder "${folder}_fail_$(date +%s)"
		skip="True"
	fi
done
