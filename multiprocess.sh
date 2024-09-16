#!/bin/bash



n=$1 # Number of intervals
m=$2 # Number of processes started
dir=$3 # name of directory containing merge.py in agent_code
command="$4 &"
echo $command

merge_command="python ./agent_code/$dir/merge.py $3"
# exit 0

# mark for multiprocessing
eval "touch ./agent_code/$dir/mp/mp.hky"

for ((i=0; i<n; i++))
do
    echo "Starting intervall $i"
    for ((j=1; j<=m; j++))
    do
        echo "Run #$j:"
        eval $command
    done
    wait
    echo "Finished interval"
    echo "Merging data"
    echo $merge_command
    eval $merge_command
    echo "Finished merging"
    eval "rm -v ./agent_code/$dir/mp/data/*"

done

echo "finished all processes"

# remove mark for multiprocessing
eval "rm ./agent_code/$dir/mp/mp.hky"
