#!/bin/sh

cd scripts

FILES="
run_rfawg.py
"
for i in 1 2 3 4 5 6
do
	for f in $FILES
	do
		echo $f $i
		python3 $f --dir 20210223_awg_synopsis
	done
done
