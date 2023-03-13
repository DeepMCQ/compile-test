#!/bin/bash
# to execute: ./run_all.sh
# !!!!! USE conda BASE environment !!!!!
set -e
set -o pipefail
rm -rf build_template
rm -rf auto_tune_bloom
rm -rf bfsg.*
rm -rf song_result.txt
rm -rf song
rm -rf org_graph
rm -rf graph
rm -rf build_graph
rm -rf trans_graph


PQ_SIZE=26

./generate_template.sh
./fill_parameters.sh $PQ_SIZE 384 ip

# use python generate_data.py to create dump_database
DATABASE_LENGTH="$(python generate_data.py)"
# DATABASE_LENGTH=231104

# build_graph
cd hnsw/examples/cpp
make all
cd ../../../
cp hnsw/examples/cpp/build_graph ./
cp hnsw/examples/cpp/trans_graph ./

mkdir org_graph
mkdir graph
./build_graph
./trans_graph


./build_graph.sh database $DATABASE_LENGTH 384 ip

cp graph/sub_graph0.bin bfsg.graph

./test_query.sh query $DATABASE_LENGTH 384 ip 1 >> song_result.txt

python cal_recall.py
