#include <bits/stdc++.h>
#include "../../hnswlib/hnswlib.h"

// const std::string hnsw_file = "hnsw.bin";
// const std::string song_file = "song.bin";

// std::ifstream input(hnsw_file, std::ios::binary);
// std::ofstream output(song_file, std::ios::binary);


// size_t offsetLevel0_;

int main(){
    int dim = 384;               // Dimension of the elements
    int max_elements = 231104;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    alg_hnsw->loadIndex("./org_graph/hnsw_200.bin", &space);

    int maxlevel_ = alg_hnsw->maxlevel_;
    size_t cur_element_count = alg_hnsw->cur_element_count;

    int vertex_offset_shift = 5;
    int num_vertices = cur_element_count;

    std::cout<<"maxlevel_ ="<<maxlevel_<<std::endl;
    std::cout<<"cur_element_count ="<<cur_element_count<<std::endl;
    std::cout<<"vertex_offset_shift ="<<vertex_offset_shift<<std::endl;



    // inline char *getDataByInternalId(tableint internal_id) const {
    //     return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    // }

    double * query = new double[dim];
    alg_hnsw->searchKnn(query, 1);

    for (int level = maxlevel_; level >= 0; level--){
        std::string subgraph_name = std::string("./graph/sub_graph")+char('0'+level)+".bin";
        FILE* fp = fopen(subgraph_name.c_str(),"wb");

        std::vector<size_t> edges;
        edges.resize(num_vertices << vertex_offset_shift);

        for (unsigned int i=0;i<num_vertices;++i) {
            size_t offset = i << vertex_offset_shift;
            if ( alg_hnsw->element_levels_[i] <level )
            {
                edges[offset] = 0;
                continue;
            }
            unsigned int *data;
            data = (unsigned int *) alg_hnsw->get_linklist_at_level(i, level);
            int size = alg_hnsw->getListCount(data);
            edges[offset] = size;
            
            if ( ( level == 0 )  && ( i==0 ) )
                printf("degree = %d\n",size);
            for (int j=0;j<size;++j){
                if ( ( level == 0 )  && ( i==0 ) )
                    printf(" %d ",data[j+1]);
                edges[ offset + j+1 ] = data[j+1];
            }
            if ( ( level == 0 )  && ( i==0 ) )
                puts("");
        }
        fwrite(&edges[0],sizeof(edges[0]) * (((size_t)num_vertices) << vertex_offset_shift),1,fp);
        fclose(fp);
    }
    return 0;
}
