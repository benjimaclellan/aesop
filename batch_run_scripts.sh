#!/bin/sh


FILES="
run_pulse_rep.py
"

cd scripts

for i in 1 2 3 4 5 6 7
do
	for f in $FILES
	do
		echo $f $i
		python3 $f --dir 20210222_pulse_reps_synopsis
	done
done
