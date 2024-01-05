#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <unistd.h>

#define ull unsigned long long

unsigned char* image_src, *image_result;

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
__m128i mullo_epi8(__m128i a, __m128i b)
{
    // unpack and multiply
    __m128i dst_even = _mm_mullo_epi16(a, b);
    __m128i dst_odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
    // repack
#ifdef __AVX2__
    // only faster if have access to VPBROADCASTW
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_and_si128(dst_even, _mm_set1_epi16(0xFF)));
#else
    return _mm_or_si128(_mm_slli_epi16(dst_odd, 8), _mm_srli_epi16(_mm_slli_epi16(dst_even,8), 8));
#endif
}

void kmeans(unsigned char* image_src, unsigned char* image_result, unsigned height, unsigned width, unsigned channels, unsigned num_cluster) {

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ull threadNum = CPU_COUNT(&cpuset);
    printf("threadNum: %d\n", threadNum);

    int *pt_cluster = (int*) malloc(height * width * sizeof(int));
    // char *centroid = (char*) malloc(channels * num_cluster * sizeof(char));
    // char *new_centroid = (char*) malloc(channels * num_cluster * sizeof(char));
    int num_cluster_4 = num_cluster + (num_cluster % 4);
    unsigned char *centroid = (unsigned char*) malloc(channels * num_cluster_4 * sizeof(unsigned char));
    unsigned char *new_centroid = (unsigned char*) malloc(channels * num_cluster_4 * sizeof(unsigned char));

    int *sum_dist = (int*) malloc(channels * num_cluster * sizeof(int));
    int *num_pt_cluster = (int*) malloc(num_cluster * sizeof(int));
    // unsigned char val[3];
    int dist, min_dist, idx, sum_val;

    // get random center
    //todo: pragma omp atomic read /write
    //#pragma omp parallel for num_threads(threadNum) -- data race: rand()
    for(int i = 0; i < num_cluster; i++) {
        int idx_i = rand() % width;
        int idx_j = rand() % height;
        centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
        centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
        centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
    }
// #pragma omp parallel for num_threads(threadNum)
// for(int i = 0; i < num_cluster; i++) {
//     unsigned int seed = omp_get_thread_num() + 1;  // Unique seed for each thread
//     int idx_i = rand_r(&seed) % width;
//     int idx_j = rand_r(&seed) % height;
//     centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
//     centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
//     centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
// }

    while (1)
    {        
        #pragma omp parallel for collapse(2) num_threads(threadNum)
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                // printf("i: %d, j: %d\n", i, j);
                __m128i val_r = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 0] ) ;
                __m128i val_g = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 1] ) ;
                __m128i val_b = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 2] ) ;

                // printf("image_src (r,g,b): %u, %u, %u \n", image_src[channels * (j + i * width) + 0], image_src[channels * (j + i * width) + 1], image_src[channels * (j + i * width) + 2]);
                // fflush(stdout);
                int min_dist = 1000000;
                int idx = 0;
                
                int k;
                for( k = 0; k < num_cluster; k+=4) {
                    // printf("centroid_r unsign char: %u, %u, %u, %u \n", centroid[(channels * k )+0], centroid[(channels * k )+3], centroid[(channels * k )+6], centroid[(channels * k )+9]);
                    // fflush(stdout);
                    
                    // printf("centroid_r (int): %d, %d, %d, %d \n", (int)centroid[(channels * k )+0], (int)centroid[(channels * k )+3], (int)centroid[(channels * k )+6], (int)centroid[(channels * k )+9] );
                    // fflush(stdout);

                    //todo: centroid 要反過來做？
                    __m128i centroid_r = _mm_set_epi32 ( (int)centroid[(channels * k )+9], (int)centroid[(channels * k )+6], (int)centroid[(channels * k )+3], (int)centroid[(channels * k )+0]);

                    // int tmp[4];
                    // _mm_storeu_si128((__m128i*)tmp, centroid_r);
                    // printf("centroid_r int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);

                    __m128i centroid_g = _mm_set_epi32 ( (int)centroid[ (channels * k )+10], (int)centroid[ (channels * k )+7], (int)centroid[(channels * k )+4], (int)centroid[  (channels * k )+1] ) ;
                    __m128i centroid_b = _mm_set_epi32 ( (int)centroid[(channels * k )+11], (int)centroid[ (channels * k )+8], (int)centroid[(channels * k )+5], (int)centroid[ (channels * k )+2] ) ;
                    
                    __m128i result_sub_r = _mm_sub_epi32(val_r, centroid_r);
                    // _mm_storeu_si128((__m128i*)tmp, result_sub_r);
                    // printf("result_sub_r int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);

                    __m128i result_square_r = _mm_mullo_epi32(result_sub_r, result_sub_r);

                    // _mm_storeu_si128((__m128i*)tmp, result_square_r);
                    // printf("result_square_r int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);

                    __m128i result_sub_g = _mm_sub_epi32(val_g, centroid_g);
                    __m128i result_square_g = _mm_mullo_epi32(result_sub_g, result_sub_g);

                    // _mm_storeu_si128((__m128i*)tmp, result_square_g);
                    // printf("result_square_g int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);

                    __m128i result_sub_b = _mm_sub_epi32(val_b, centroid_b);
                    __m128i result_square_b = _mm_mullo_epi32(result_sub_b, result_sub_b);


                    // _mm_storeu_si128((__m128i*)tmp, result_square_b);
                    // printf("result_square_b int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);

                    __m128i dist_128 = _mm_add_epi32(result_square_r, result_square_g);
                    dist_128 = _mm_add_epi32(dist_128, result_square_b);

                    // _mm_storeu_si128((__m128i*)tmp, dist_128);
                    // printf("dist_128 int: %d, %d, %d, %d \n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    // fflush(stdout);
                    // todo 有誤差
                     // Convert integers to single-precision floating-point
                    __m128 float_dist_128 = _mm_cvtepi32_ps(dist_128);
                    // Compute square root
                    __m128 sqrt_dist_128 = _mm_sqrt_ps(float_dist_128);
                    // Convert the result back to integers
                    __m128i int_sqrt_result = _mm_cvtps_epi32(sqrt_dist_128);
                    
                    //_mm_storeu_si128((__m128i*)tmp, int_sqrt_result);
                    //printf("int_sqrt_result int: %d, %d, %d, %d \n\n\n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    //fflush(stdout);

                    // sleep(10);
                    int dist[4];
                    _mm_storeu_si128((__m128i*)dist, int_sqrt_result);
                    
                    for (int ii = 0; k+ii < num_cluster && ii <4 ; ++ii) {
                        if (dist[ii] < min_dist) {
                            min_dist = dist[ii];
                            idx = k+ii;
                            // printf("iidx: %d \n", idx);
                            // fflush(stdout);
                        }
                    }
                }
                // printf("idx: %d \n\n", idx);
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

        // todo 2: data race: idx, idx declare within the scope
        #pragma omp parallel for collapse(2) // -- data race: idx , sum_dist
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                
                int idx = pt_cluster[j + i * width];
                // SSE: idx 有data race
                num_pt_cluster[idx] += 1;
                sum_dist[idx * channels + 0] += image_src[channels * (j + i * width) + 0];
                sum_dist[idx * channels + 1] += image_src[channels * (j + i * width) + 1];
                sum_dist[idx * channels + 2] += image_src[channels * (j + i * width) + 2];
            }
        }

        #pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < num_cluster; i++) {
            if(num_pt_cluster[i] != 0) {
                new_centroid[channels * i + 0] = sum_dist[i * channels + 0] / num_pt_cluster[i];
                new_centroid[channels * i + 1] = sum_dist[i * channels + 1] / num_pt_cluster[i];
                new_centroid[channels * i + 2] = sum_dist[i * channels + 2] / num_pt_cluster[i];
            }
        }

        sum_val = 0;
        #pragma omp parallel for num_threads(threadNum) reduction(+:sum_val)
        for(int i = 0; i < num_cluster; i++) { 
            int dist = 0;
            dist += (new_centroid[channels * i + 0] - centroid[channels * i + 0]) * (new_centroid[channels * i + 0] - centroid[channels * i + 0]);
            dist += (new_centroid[channels * i + 1] - centroid[channels * i + 1]) * (new_centroid[channels * i + 1] - centroid[channels * i + 1]);
            dist += (new_centroid[channels * i + 2] - centroid[channels * i + 2]) * (new_centroid[channels * i + 2] - centroid[channels * i + 2]);
        
            sum_val += sqrt(dist);
            centroid[channels * i + 0] = new_centroid[channels * i + 0];
            centroid[channels * i + 1] = new_centroid[channels * i + 1];
            centroid[channels * i + 2] = new_centroid[channels * i + 2];
            
        }
        // printf("sum_val: %d \n", sum_val);
        fflush(stdout);
        if(sum_val < num_cluster * 4.1) {
            break;
        }
    }
    //todo 3: add #pragma omp atomic for if clause to avoid dist data race
    #pragma omp parallel for collapse(2) num_threads(threadNum)
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            unsigned char val[3];
            __m128i val_r = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 0] ) ;
            __m128i val_g = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 1] ) ;
            __m128i val_b = _mm_set1_epi32 ( (int)image_src[channels * (j + i * width) + 2] ) ;

            int min_dist = 1000000;
            int dist, idx;

            for(int k = 0; k < num_cluster; k++) {
                __m128i centroid_r = _mm_set_epi32 ( (int)centroid[(channels * k )+9], (int)centroid[(channels * k )+6], (int)centroid[(channels * k )+3], (int)centroid[(channels * k )+0]);
                __m128i centroid_g = _mm_set_epi32 ( (int)centroid[ (channels * k )+10], (int)centroid[ (channels * k )+7], (int)centroid[(channels * k )+4], (int)centroid[  (channels * k )+1] ) ;
                __m128i centroid_b = _mm_set_epi32 ( (int)centroid[(channels * k )+11], (int)centroid[ (channels * k )+8], (int)centroid[(channels * k )+5], (int)centroid[ (channels * k )+2] ) ;

                __m128i result_sub_r = _mm_sub_epi32(val_r, centroid_r);
                __m128i result_square_r = _mm_mullo_epi32(result_sub_r, result_sub_r);

                __m128i result_sub_g = _mm_sub_epi32(val_g, centroid_g);
                __m128i result_square_g = _mm_mullo_epi32(result_sub_g, result_sub_g);

                __m128i result_sub_b = _mm_sub_epi32(val_b, centroid_b);
                __m128i result_square_b = _mm_mullo_epi32(result_sub_b, result_sub_b);

                __m128i dist_128 = _mm_add_epi32(result_square_r, result_square_g);
                dist_128 = _mm_add_epi32(dist_128, result_square_b);

                // Convert integers to single-precision floating-point
                __m128 float_dist_128 = _mm_cvtepi32_ps(dist_128);
                __m128 sqrt_dist_128 = _mm_sqrt_ps(float_dist_128);
                // Convert the result back to integers
                __m128i int_sqrt_result = _mm_cvtps_epi32(sqrt_dist_128);
                
                
                int dist[4];
                _mm_storeu_si128((__m128i*)dist, int_sqrt_result);
                
                for (int ii = 0; k+ii < num_cluster && ii <4 ; ++ii) {
                    if (dist[ii] < min_dist) {
                        min_dist = dist[ii];
                        idx = k+ii;
                    }
                }
                
            }

            image_result[channels * (j + i * width) + 0] = centroid[channels * idx + 0];
            image_result[channels * (j + i * width) + 1] = centroid[channels * idx + 1];
            image_result[channels * (j + i * width) + 2] = centroid[channels * idx + 2];
        }
    }
}

int main(int argc, char** argv) {
    double total_time;
	timespec total_time1, total_time2;

    srand(time(0));

    assert(argc == 3);
    unsigned height, width, channels;
    image_src = NULL;
    read_png(argv[1], &image_src, &height, &width, &channels);
    image_result = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    clock_gettime(CLOCK_MONOTONIC, &total_time1);
    kmeans(image_src, image_result, height, width, channels, 10);
    clock_gettime(CLOCK_MONOTONIC, &total_time2);
	
    total_time = cal_time(total_time1, total_time2);
    printf(" total_time:  %.5f\n", total_time);

    write_png(argv[2], image_result, height, width, channels);
   

    return 0;
}
