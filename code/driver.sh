#!/usr/bin/env bash

resd="../data/results/"
dset="../data/aggregation_connect+demo_dset25x2x2x20.h5"

# exps="mca"  # For D100
exps="mca session subsample" # For D25
nmca="20 15 10 5 2" 
targets="bmi age cholesterol sex rel_vo2max"
aggs="ref meta mega mean median consensus"
classifs="SVM LR RF"

for t in ${targets}
do
  for c in ${classifs}
  do
    for e in ${exps}
    do
      if [[ $e != "mca" ]]
      then
        agg_iter=`echo ${aggs} | cut -d " " -f -4`
      else
        agg_iter=${aggs}
      fi
      for a in ${agg_iter}
      do
        bstr="$e $t $a $c"
        if [[ $e == "mca" ]]
        then
          for n in ${nmca}
          do
            python wrapper.py ${resd} ${dset} ${e} ${t} ${a} ${c} --n_mca ${n} --verbose
          done
        else
          python wrapper.py ${resd} ${dset} ${e} ${t} ${a} ${c} --verbose
        fi
      done
    done
  done
done

