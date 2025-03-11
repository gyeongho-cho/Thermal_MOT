#!/bin/bash

# track_thresh 값 배열
t_thresh=0.3
m_thresh=0.99
# 시퀀스 번호 배열
seq_list=(0 1 2 3 4 5)

for seq in "${seq_list[@]}"
do
    echo "Running test.py --track_thresh $t_thresh --seq $seq --match_thresh $m_thresh" 
    python test.py --track_thresh $t_thresh --seq $seq --match_thresh $m_thresh
done