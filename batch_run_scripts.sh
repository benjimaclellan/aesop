#!/bin/sh


FILES="
batch_pulse_rep.py
batch_rfawg.py
"
cd scripts

for i in 1 2
do
	for f in $FILES
	do
		echo $f $i
		python3 $f --dir 20210219_evolution_tree
	done
done
