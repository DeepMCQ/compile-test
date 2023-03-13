#include "../../hnswlib/hnswlib.h"

typedef float value_t;
typedef size_t idx_t;


#include<bits/stdc++.h>
#include "parser.h"



std::vector<std::vector<std::pair<int,value_t>>> batch_queries;

void query_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
    batch_queries.push_back(point);
	// Uncomment the following lines to have a finer granularity batch processing
    //if(batch_queries.size() == ACC_BATCH_SIZE){
    //    flush_queries();
    //}
	/////////////////////
}


int main() {
    int dim = 384;               // Dimension of the elements
    int max_elements = 231104;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    float* data = new float[dim * max_elements];
    FILE* fp = fopen("database","rb");
    fread(data,sizeof(float) * (dim * max_elements),1,fp);

    for (int i =0;i<10;++i)
        printf("%.6f ",*(data+i));
    puts("");

    bool flag_build = true;
    bool flag_query = true;
    if (flag_build){
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(data + i * dim, i);
            if ( i %20000 ==0 )
                printf("add %d\n",i);
        }
    printf("addPoint finished\n");
    fclose(fp);

    alg_hnsw->saveIndex("./org_graph/hnsw_200.bin");
    }
    else{
        alg_hnsw->loadIndex("./org_graph/hnsw_200.bin", &space);
    }

    if (flag_query) {
        std::unique_ptr<Parser> query_parser(new Parser("query",query_callback));
        int num_query = batch_queries.size();

        float* data = new float[dim * num_query];
        for (int i=0;i<dim*num_query;i++)
            data[i] = batch_queries[i/dim][i%dim].second;

        FILE* result_file = fopen("result.txt","w+");
        for (int i=0;i<num_query;++i){
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            fprintf(result_file,"%ld\n",label);
        }
        fclose(result_file);
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
