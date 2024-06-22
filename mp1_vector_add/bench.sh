#!/bin/bash

echo "-------BENCH---------";
for i in 0 1 2 3 4 5 6 7 8 9;
do
  echo "Case $i"
	./out ./data/${i}/input0.raw ./data/${i}/input1.raw ./data/${i}/output.raw
  echo ""
done
