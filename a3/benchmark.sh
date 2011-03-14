#!/bin/bash


for MODEL in MODEL1 HEURISTIC; do
  for TRAIN_SIZE in 10 100 1000 10000; do
    java -cp assign3.jar:assign3-submit.jar -server -mx6000m edu.berkeley.nlp.assignments.assign3.AlignmentTester -path ./data -alignerType $MODEL -data test -printAlignments -maxTrain $TRAIN_SIZE | tee "log/$MODEL $TRAIN_SIZE"
  done
done

