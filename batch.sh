#!/bin/sh

cd scripts

FILES="
run_rfawg.py
run_pulse_rep.py
"
for i in 1 2 3 4
do
	for f in $FILES
	do
		echo $f $i
		python3 $f --dir 20220108_waveformgeneration
	done
done
