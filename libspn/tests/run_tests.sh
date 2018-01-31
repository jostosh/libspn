#!/bin/bash

set -e

echo "Running 10 tests"

for i in `seq 1 5`;
do
	python3 $1 --save-to $2/i7-4720HQ_GTX960_${i}.txt
done

echo "Tests completed!"
