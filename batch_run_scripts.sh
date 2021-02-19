#!/bin/sh


FILES="
#run_topology_phase_sensitivity.py
#run_topology_phase_sensitivity.py
#run_topology_pulse_rep.py
#run_topology_rfawg.py
batch_pulse_rep.py
batch_rfawg.py
"
cd scripts

for i in 1 2 3
do
	for f in $FILES
	do
		echo $f $i
		python3 $fe --dir 20210219_evolution_tree
	done
done
