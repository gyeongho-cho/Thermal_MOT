#!/bin/bash

# track_thresh 값 배열
# track_thresh_list=(0.1 0.2 0.3 0.4)
# match_thresh_list=(0.9 0.8 0.7)
track_thresh_list=(0.4)
match_thresh_list=(0.95)
# 시퀀스 번호 배열
seq_list=(0 1 2)

for t_thresh in "${track_thresh_list[@]}"
do
    for m_thresh in "${match_thresh_list[@]}"
    do
        for seq in "${seq_list[@]}"
        do
            echo "Running test.py --track_thresh $t_thresh --seq $seq --match_thresh $m_thresh" 
            python test.py --track_thresh $t_thresh --seq $seq --match_thresh $m_thresh --track_buffer 50
        done
    done
done