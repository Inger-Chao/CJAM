#!/bin/bash

for ((i=1; i<=80; i++))
do
python3 test.py --iter=$((i*1000)) >> acc.txt
done
