运行环境：

`CUDA 11.4, gcc 9.2.0, PyTorch 1.12, torch-vision 0.14.0`

运行 `run_all.sh` 完成评测

具体命令解释：

```shell
#!/bin/bash
# to execute: ./run_all.sh
# !!!!! USE conda BASE environment !!!!!
# 清理上次生成数据
set -e
set -o pipefail
rm -rf build_template
rm -rf auto_tune_bloom
rm -rf bfsg.*
rm -rf song_result.txt
# rm -rf dump_*
rm -rf song
rm -rf org_graph
rm -rf graph

# 搜索参数，越大 recall 越高，速度越慢
PQ_SIZE=26

# 编译 SONG 主程序
./generate_template.sh
./fill_parameters.sh $PQ_SIZE 384 ip

# 生成假数据，并获得 database 大小
# use python generate_data.py to create dump_database
DATABASE_LENGTH="$(python generate_data.py)"
# DATABASE_LENGTH=231104

# 编译用于生成 HNSW 图的主程序 build_graph 和 trans_graph
cd hnsw/examples/cpp
make all
cd ../../../
cp hnsw/examples/cpp/build_graph ./
cp hnsw/examples/cpp/trans_graph ./

# 运行程序生成 HNSW 图
mkdir org_graph
mkdir graph
./build_graph
./trans_graph

# 生成检索中图的必要信息
./build_graph.sh database $DATABASE_LENGTH 384 ip

cp graph/sub_graph0.bin bfsg.graph


# 进行查询
./test_query.sh query $DATABASE_LENGTH 384 ip 1 >> song_result.txt

# 测试 recall
python cal_recall.py
```
