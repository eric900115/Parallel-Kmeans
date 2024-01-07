#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
//#include <mpi.h>
#define ull unsigned long long
#define CENTROID_TAG 0
#define UPDATE_CENTROID_TAG 1
#define UPDATE_IMAGE_TAG 2

unsigned char* image_src, *image_result;
int rank, size; 


double cal_time(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0)
	{
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else
	{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp.tv_sec + (double)temp.tv_nsec / 1000000000.0;
}

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];;
    FILE* infile;
    infile = fopen(filename, "rb");

    if (infile == NULL) {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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

void kmeans(unsigned char* image_src, unsigned char* image_result, unsigned height, unsigned width, unsigned channels, unsigned num_cluster) {

    //printf("rank: %d\n", rank);

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ull threadNum = CPU_COUNT(&cpuset);

    int *pt_cluster = (int*) malloc(height * width * sizeof(int));
    // char *centroid = (char*) malloc(channels * num_cluster * sizeof(char));
    // char *new_centroid = (char*) malloc(channels * num_cluster * sizeof(char));

    unsigned char *centroid = (unsigned char*) malloc(channels * num_cluster * sizeof(unsigned char));
    unsigned char *new_centroid = (unsigned char*) malloc(channels * num_cluster * sizeof(unsigned char));

    unsigned char *tmp_centroid = (unsigned char*) malloc(channels * num_cluster * sizeof(char));
    int *sum_dist = (int*) malloc(channels * num_cluster * sizeof(int));
    int *num_pt_cluster = (int*) malloc(num_cluster * sizeof(int));
    unsigned char* tmp_image_result = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    // unsigned char val[3];
    int dist, min_dist, idx, sum_val;
    MPI_Status status;

    // get random center
    // if(rank == 0){   
    //     for(int i = 0; i < num_cluster; i++) {
    //         int idx_i = rand() % width;
    //         int idx_j = rand() % height;
    //         centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
    //         centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
    //         centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
    //     }
    // }
    // MPI_Bcast(centroid, channels * num_cluster, MPI_CHAR, 0, MPI_COMM_WORLD);

    for(int i = 0; i < num_cluster; i++) {
        int idx_i = rand() % width;
        int idx_j = rand() % height;
        centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
        centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
        centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    

    while (1)
    {
        #pragma omp parallel for collapse(2) num_threads(threadNum)
        for(int i = rank; i < height; i+=size) {
            for(int j = 0; j < width; j++) {
                unsigned char val[3];
                val[0] = image_src[channels * (j + i * width) + 0];
                val[1] = image_src[channels * (j + i * width) + 1];
                val[2] = image_src[channels * (j + i * width) + 2];
                int min_dist = 1000000;
                int dist, idx;
                for(int k = 0; k < num_cluster; k++) {
                    dist = 0;
                    dist += (val[0] - centroid[channels * k + 0]) * (val[0] - centroid[channels * k + 0]);
                    dist += (val[1] - centroid[channels * k + 1]) * (val[1] - centroid[channels * k + 1]);
                    dist += (val[2] - centroid[channels * k + 2]) * (val[2] - centroid[channels * k + 2]);
                    dist = sqrt(dist);
                    if(dist < min_dist) {
                        min_dist = dist;
                        idx = k;
                    }
                }
                pt_cluster[j + i * width] = idx;
            }
        }

        // clear the value
        #pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < num_cluster; i++) {
            num_pt_cluster[i] = 0;
            sum_dist[0 + i * channels] = 0;
            sum_dist[1 + i * channels] = 0;
            sum_dist[2 + i * channels] = 0;
        }

        //#pragma omp parallel for collapse(2) -- data race: idx 

        for(int i = rank; i < height; i+=size) {
            for(int j = 0; j < width; j++) {
                idx = pt_cluster[j + i * width]; // get cluster id
                num_pt_cluster[idx] += 1;
                sum_dist[idx * channels + 0] += image_src[channels * (j + i * width) + 0];
                sum_dist[idx * channels + 1] += image_src[channels * (j + i * width) + 1];
                sum_dist[idx * channels + 2] += image_src[channels * (j + i * width) + 2];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, sum_dist, channels * num_cluster, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, num_pt_cluster, num_cluster, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        //MPI_Barrier(MPI_COMM_WORLD);


        //todo: MPI 要merge [num_pt_cluster] and [sum_dist]
                   
        #pragma omp parallel for num_threads(threadNum)
        for(int i = rank; i < num_cluster; i+=size) {
            if(num_pt_cluster[i] != 0) {
                new_centroid[channels * i + 0] = sum_dist[i * channels + 0] / num_pt_cluster[i];
                new_centroid[channels * i + 1] = sum_dist[i * channels + 1] / num_pt_cluster[i];
                new_centroid[channels * i + 2] = sum_dist[i * channels + 2] / num_pt_cluster[i];
            }
        }
        
        //todo : scatter new_centroid from rank 0.

        // calculate the sum of difference between old center and new center
        // Also, save the centroid value
        sum_val = 0;
        int globa_sum_val = 0;
        #pragma omp parallel for num_threads(threadNum) reduction(+:sum_val)
        for(int i = rank; i < num_cluster; i+=size) {
            int dist = 0;
            dist += (new_centroid[channels * i + 0] - centroid[channels * i + 0]) * (new_centroid[channels * i + 0] - centroid[channels * i + 0]);
            dist += (new_centroid[channels * i + 1] - centroid[channels * i + 1]) * (new_centroid[channels * i + 1] - centroid[channels * i + 1]);
            dist += (new_centroid[channels * i + 2] - centroid[channels * i + 2]) * (new_centroid[channels * i + 2] - centroid[channels * i + 2]);
             
            sum_val += sqrt(dist);
            centroid[channels * i + 0] = new_centroid[channels * i + 0];
            centroid[channels * i + 1] = new_centroid[channels * i + 1];
            centroid[channels * i + 2] = new_centroid[channels * i + 2];
        }
        for (size_t process = 0; process < size; process++)
        {
            if(process != rank){
                MPI_Send(centroid, channels*num_cluster, MPI_CHAR, process, UPDATE_CENTROID_TAG, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (size_t process = 0; process < size-1; process++)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int MPI_SOURCE = status.MPI_SOURCE;
            int MPI_TAG = status.MPI_TAG;
            MPI_Recv( tmp_centroid ,channels* num_cluster, MPI_CHAR, MPI_SOURCE, UPDATE_CENTROID_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t i = MPI_SOURCE; i < num_cluster; i+=size)
            {
                centroid[channels * i + 0] = tmp_centroid[channels * i + 0];
                centroid[channels * i + 1] = tmp_centroid[channels * i + 1];
                centroid[channels * i + 2] = tmp_centroid[channels * i + 2];
            }
        }
        #pragma omp critical
        globa_sum_val += sum_val;

        if(globa_sum_val < num_cluster * 4.1) {
            break;
        }
        #pragma omp barrier
        globa_sum_val = 0;
        #pragma omp barrier
    }
    // write image 
    #pragma omp parallel for collapse(2) num_threads(threadNum)
    for(int i = rank; i < height; i+=size) {
        for(int j = 0; j < width; j++) {
            unsigned char val[3];
            val[0] = image_src[channels * (j + i * width) + 0];
            val[1] = image_src[channels * (j + i * width) + 1];
            val[2] = image_src[channels * (j + i * width) + 2];
            int min_dist = 1000000;
            int dist, idx;
            for(int k = 0; k < num_cluster; k++) {
                // calculate l2 norm
                dist = 0;
                dist += (val[0] - centroid[channels * k + 0]) * (val[0] - centroid[channels * k + 0]);
                dist += (val[1] - centroid[channels * k + 1]) * (val[1] - centroid[channels * k + 1]);
                dist += (val[2] - centroid[channels * k + 2]) * (val[2] - centroid[channels * k + 2]);
                dist = sqrt(dist);
                if(dist < min_dist) {
                    min_dist = dist;
                    idx = k;
                }
            }

            image_result[channels * (j + i * width) + 0] = centroid[channels * idx + 0];
            image_result[channels * (j + i * width) + 1] = centroid[channels * idx + 1];
            image_result[channels * (j + i * width) + 2] = centroid[channels * idx + 2];
        }
    }
    // write image 
    if(rank != 0){
        MPI_Send(image_result, channels*height*width, MPI_UNSIGNED_CHAR, 0, UPDATE_IMAGE_TAG, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        for (size_t count = 0; count < size-1; count++)
        {
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int MPI_SOURCE = status.MPI_SOURCE;
            int MPI_TAG = status.MPI_TAG;
            MPI_Recv(tmp_image_result ,channels*height*width, MPI_UNSIGNED_CHAR, MPI_SOURCE, UPDATE_IMAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t i = MPI_SOURCE; i < height; i+=size)
            {
                for(int j = 0; j < width; j++) {
                    image_result[channels * (j + i * width) + 0] = tmp_image_result[channels * (j + i * width) + 0];
                    image_result[channels * (j + i * width) + 1] = tmp_image_result[channels * (j + i * width) + 1];
                    image_result[channels * (j + i * width) + 2] = tmp_image_result[channels * (j + i * width) + 2];
                }
            }
        }
    }
}



int main(int argc, char** argv) {
    double total_time;
	timespec total_time1, total_time2;

    srand(time(0));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    assert(argc == 3);
    unsigned height, width, channels;
    image_src = NULL;

    read_png(argv[1], &image_src, &height, &width, &channels);
    
    image_result = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    clock_gettime(CLOCK_MONOTONIC, &total_time1);
    kmeans(image_src, image_result, height, width, channels, 128);

    clock_gettime(CLOCK_MONOTONIC, &total_time2);
	
    total_time = cal_time(total_time1, total_time2);
    if(rank ==0 ){
        printf(" total_time:  %.5f\n", total_time);
    }
    //printf(" total_time:  %.5f\n", total_time);
// todo rank ==0/
    if(rank == 0){
        write_png(argv[2], image_result, height, width, channels);
    }
    MPI_Finalize();

    
    return 0;
}
