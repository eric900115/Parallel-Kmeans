#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <math.h>

#define num_cluster 10
#define INF 1000000
#define BlockFactor 30

unsigned char *image_src, *image_result;
unsigned char *device_image_src, *device_image_result;

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void KmeansAssignCluster(unsigned char* image_src, unsigned char* centroid, int* num_pt_cluster,\
                                     int* sum_dist, int height, int width, int channels) {
    
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int min_dist = INF;
    int dist = 0;
    int cluster_idx;

    if(!(bound_check(idx_x, 0, width) && bound_check(idx_y, 0, height))) {
        return;
    }

    for(int k = 0; k < num_cluster; k++) {
        // calculate l2 norm
        dist = 0;
        dist += (image_src[channels * (idx_x + idx_y * width) + 0] - centroid[channels * k + 0]) * (image_src[channels * (idx_x + idx_y * width) + 0] - centroid[channels * k + 0]);
        dist += (image_src[channels * (idx_x + idx_y * width) + 1] - centroid[channels * k + 1]) * (image_src[channels * (idx_x + idx_y * width) + 1] - centroid[channels * k + 1]);
        dist += (image_src[channels * (idx_x + idx_y * width) + 2] - centroid[channels * k + 2]) * (image_src[channels * (idx_x + idx_y * width) + 2] - centroid[channels * k + 2]);
        dist = sqrt((float)dist);

        if(dist < min_dist) {
            min_dist = dist;
            cluster_idx = k;
        }
    }

    atomicAdd(&num_pt_cluster[cluster_idx], 1);
    atomicAdd(&sum_dist[cluster_idx * channels + 0], image_src[channels * (idx_x + idx_y * width) + 0]);
    atomicAdd(&sum_dist[cluster_idx * channels + 1], image_src[channels * (idx_x + idx_y * width) + 1]);
    atomicAdd(&sum_dist[cluster_idx * channels + 2], image_src[channels * (idx_x + idx_y * width) + 2]);
}

__global__ void KmeansUpdateCentroid(unsigned char* new_centroid, int* sum_dist, int* num_pt_cluster, unsigned int channels) {

    const int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    //int num_pt = num_pt_cluster[cluster_idx];

    new_centroid[channels * cluster_idx + 0] = sum_dist[channels * cluster_idx + 0] / num_pt_cluster[cluster_idx];
    new_centroid[channels * cluster_idx + 1] = sum_dist[channels * cluster_idx + 1] / num_pt_cluster[cluster_idx];
    new_centroid[channels * cluster_idx + 2] = sum_dist[channels * cluster_idx + 2] / num_pt_cluster[cluster_idx];
}

__global__ void KmeansCalLoss(unsigned char* new_centroid, unsigned char* centroid, int* sum_loss, unsigned int channels) {

    const int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int dist = 0;

    dist += (new_centroid[channels * cluster_idx + 0] - centroid[channels * cluster_idx + 0]) * (new_centroid[channels * cluster_idx + 0] - centroid[channels * cluster_idx + 0]);
    dist += (new_centroid[channels * cluster_idx + 1] - centroid[channels * cluster_idx + 1]) * (new_centroid[channels * cluster_idx + 1] - centroid[channels * cluster_idx + 1]);
    dist += (new_centroid[channels * cluster_idx + 2] - centroid[channels * cluster_idx + 2]) * (new_centroid[channels * cluster_idx + 2] - centroid[channels * cluster_idx + 2]);
    dist += sqrt((float)dist);

    atomicAdd(sum_loss, dist);
}

__global__ void AssignCenter(unsigned char* new_centroid, unsigned char* centroid, unsigned channels) {

    const int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    new_centroid[channels * cluster_idx + 0] = centroid[channels * cluster_idx + 0];
    new_centroid[channels * cluster_idx + 1] = centroid[channels * cluster_idx + 1];
    new_centroid[channels * cluster_idx + 2] = centroid[channels * cluster_idx + 2];

}

__global__ void KmeansWriteResult(unsigned char* image_src, unsigned char* image_result, \
            unsigned char* centroid, unsigned width, unsigned height, unsigned channels) {

    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int min_dist = INF;
    int dist = 0;
    int cluster_idx;

    if(!(bound_check(idx_x, 0, width) && bound_check(idx_y, 0, height))) {
        return;
    }
   
    for(int k = 0; k < num_cluster; k++) {
        // calculate l2 norm
        dist = 0;
        dist += (image_src[channels * (idx_x + idx_y * width) + 0] - centroid[channels * k + 0]) * (image_src[channels * (idx_x + idx_y * width) + 0] - centroid[channels * k + 0]);
        dist += (image_src[channels * (idx_x + idx_y * width) + 1] - centroid[channels * k + 1]) * (image_src[channels * (idx_x + idx_y * width) + 1] - centroid[channels * k + 1]);
        dist += (image_src[channels * (idx_x + idx_y * width) + 2] - centroid[channels * k + 2]) * (image_src[channels * (idx_x + idx_y * width) + 2] - centroid[channels * k + 2]);
        dist = sqrt((float)dist);
        if(dist < min_dist) {
            min_dist = dist;
            cluster_idx = k;
        }
    }

    image_result[channels * (idx_x + idx_y * width) + 0] = centroid[channels * cluster_idx + 0];
    image_result[channels * (idx_x + idx_y * width) + 1] = centroid[channels * cluster_idx + 1];
    image_result[channels * (idx_x + idx_y * width) + 2] = centroid[channels * cluster_idx + 2];
}

void kmeans(unsigned height, unsigned width, unsigned channels) {

    unsigned char *centroid = (unsigned char*) malloc(channels * num_cluster * sizeof(char));

    //unsigned char val[3];
    int dist, min_dist, idx, sum_loss;

    int *device_pt_cluster;
    int *device_sum_dist;
    int *device_num_pt_cluster;
    int *device_sum_loss;
    unsigned char *device_centroid, *device_new_centroid;

    cudaMalloc(&device_sum_loss, 1 * sizeof(int));
    cudaMalloc(&device_pt_cluster, height * width * sizeof(int));
    cudaMalloc(&device_sum_dist, channels * num_cluster * sizeof(int));
    cudaMalloc(&device_num_pt_cluster, num_cluster * sizeof(int));
    cudaMalloc(&device_centroid, channels * num_cluster * sizeof(unsigned char));
    cudaMalloc(&device_new_centroid, channels * num_cluster * sizeof(unsigned char));

    dim3 num_blocks(width / BlockFactor + 1, height / BlockFactor + 1);
    dim3 num_threads(BlockFactor, BlockFactor);

    // get random center
    for(int i = 0; i < num_cluster; i++) {
        int idx_i = rand() % width;
        int idx_j = rand() % height;
        centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
        centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
        centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
    }

    cudaMemcpy(device_centroid, centroid, channels * num_cluster * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(device_image_src, image_src, channels * height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    while (1)
    {
        // Clustering All of the Pixels in image
        KmeansAssignCluster <<<num_blocks, num_threads>>> ( \
            device_image_src, device_centroid, device_num_pt_cluster, \
            device_sum_dist, height, width, channels \
        );

        // Clear the Value
        cudaMemset(device_num_pt_cluster, 0, num_cluster * sizeof(int));
        cudaMemset(device_sum_dist, 0, channels * num_cluster * sizeof(int));

        // Update Center
        KmeansUpdateCentroid <<<1, num_cluster>>> (device_new_centroid, device_sum_dist, device_num_pt_cluster, channels);

        // calculate the sum of difference between old center and new center
        KmeansCalLoss <<<1, num_cluster>>> (device_new_centroid, device_centroid, device_sum_loss, channels);
        
        // Store the Centroid Value
        AssignCenter <<<1, num_cluster>>> (device_new_centroid, device_centroid, channels);

        cudaMemcpy(&sum_loss, device_sum_loss, 1 * sizeof(int), cudaMemcpyDeviceToHost);

        // if the sum < threshold, stop the iteraton
        if(sum_loss < num_cluster * 4.1) {
            break;
        }
    }

    KmeansWriteResult <<<num_blocks, num_threads>>> ( \
        device_image_src, device_image_result, \
        device_centroid, width, height, channels \
    );

    cudaMemcpy(image_result, device_image_result, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {

    srand(time(0));

    assert(argc == 3);
    unsigned height, width, channels;
    image_src = NULL;
    
    read_png(argv[1], &image_src, &height, &width, &channels);

    image_result = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    cudaHostRegister(image_src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    cudaMalloc(&device_image_src, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&device_image_result, height * width * channels * sizeof(unsigned char));

    kmeans(height, width, channels);

    write_png(argv[2], image_result, height, width, channels);

    free(image_src);
    free(image_result);
    cudaFree(device_image_src);
    cudaFree(device_image_result);

    return 0;
}
