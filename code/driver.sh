#!/usr/bin/env bash

resd="../data/results/"
dset="../data/aggregation_connect+demo_dset25x2x2x20.h5"
logf="../data/log_D25.txt"

mkdir -p ${resd}

# exps="mca"  # For D100
exps="mca session subsample" # For D25

# targets="bmi age cholesterol sex rel_vo2max"  # For D100
targets="bmi age sex rel_vo2max"  # For D25

nmca="20 15 10 5 2"
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
        if [[ $e == "mca" ]]
        then
          for n in ${nmca}
          do
            echo $e $t $a $c $n &>> ${logf}
            (time python wrapper.py ${resd} ${dset} ${e} ${t} ${a} ${c} --n_mca ${n} --verbose) &>>${logf}
          done
        else
          echo $e $t $a $c &>> ${logf}
          (time python wrapper.py ${resd} ${dset} ${e} ${t} ${a} ${c} --verbose) &>>${logf}
        fi
      done
    done
  done
done

