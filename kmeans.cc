#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <math.h>

unsigned char* image_src, *image_result;

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

void kmeans(unsigned char* image_src, unsigned char* image_result, unsigned height, unsigned width, unsigned channels, unsigned num_cluster) {

    int *pt_cluster = (int*) malloc(height * width * sizeof(int));
    char *centroid = (char*) malloc(channels * num_cluster * sizeof(char));
    char *new_centroid = (char*) malloc(channels * num_cluster * sizeof(char));
    int *sum_dist = (int*) malloc(channels * num_cluster * sizeof(int));
    int *num_pt_cluster = (int*) malloc(num_cluster * sizeof(int));
    char val[3];
    int dist, min_dist, idx, sum_val;

    // get random center
    for(int i = 0; i < num_cluster; i++) {
        int idx_i = rand() % width;
        int idx_j = rand() % height;
        centroid[channels * i + 0] = image_src[channels * (idx_i + idx_j * width) + 0];
        centroid[channels * i + 1] = image_src[channels * (idx_i + idx_j * width) + 1];
        centroid[channels * i + 2] = image_src[channels * (idx_i + idx_j * width) + 2];
    }

    while (1)
    {
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                
                val[0] = image_src[channels * (j + i * width) + 0];
                val[1] = image_src[channels * (j + i * width) + 1];
                val[2] = image_src[channels * (j + i * width) + 2];
                min_dist = 1000000;
                
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
                // store which cluster the data belong to
                pt_cluster[j + i * width] = idx;
            }
        }

        // clear the value
        for(int i = 0; i < num_cluster; i++) {
            num_pt_cluster[i] = 0;
            sum_dist[0 + i * channels] = 0;
            sum_dist[1 + i * channels] = 0;
            sum_dist[2 + i * channels] = 0;
        }

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                idx = pt_cluster[j + i * width]; // get cluster id
                num_pt_cluster[idx] += 1;
                sum_dist[idx * channels + 0] += image_src[channels * (j + i * width) + 0];
                sum_dist[idx * channels + 1] += image_src[channels * (j + i * width) + 1];
                sum_dist[idx * channels + 2] += image_src[channels * (j + i * width) + 2];
            }
        }

        for(int i = 0; i < num_cluster; i++) {
            if(num_pt_cluster[i] != 0) {
                new_centroid[channels * i + 0] = sum_dist[i * channels + 0] / num_pt_cluster[i];
                new_centroid[channels * i + 1] = sum_dist[i * channels + 1] / num_pt_cluster[i];
                new_centroid[channels * i + 2] = sum_dist[i * channels + 2] / num_pt_cluster[i];
            }
        }

        // calculate the sum of difference between old center and new center
        // Also, save the centroid value
        sum_val = 0;
        for(int i = 0; i < num_cluster; i++) {
            dist = 0;
            dist += (new_centroid[channels * i + 0] - centroid[channels * i + 0]) * (new_centroid[channels * i + 0] - centroid[channels * i + 0]);
            dist += (new_centroid[channels * i + 1] - centroid[channels * i + 1]) * (new_centroid[channels * i + 1] - centroid[channels * i + 1]);
            dist += (new_centroid[channels * i + 2] - centroid[channels * i + 2]) * (new_centroid[channels * i + 2] - centroid[channels * i + 2]);
            sum_val += sqrt(dist);
            
            centroid[channels * i + 0] = new_centroid[channels * i + 0];
            centroid[channels * i + 1] = new_centroid[channels * i + 1];
            centroid[channels * i + 2] = new_centroid[channels * i + 2];
        }

        // if the sum < threshold, stop the iteraton
        if(sum_val < num_cluster * 4.1) {
            break;
        }
    }

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            val[0] = image_src[channels * (j + i * width) + 0];
            val[1] = image_src[channels * (j + i * width) + 1];
            val[2] = image_src[channels * (j + i * width) + 2];
            min_dist = 1000000;
        
            for(int k = 0; k < num_cluster; k++) {
                // calculate l2 norm
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
}

int main(int argc, char** argv) {

    srand(time(0));

    assert(argc == 3);
    unsigned height, width, channels;
    image_src = NULL;
    read_png(argv[1], &image_src, &height, &width, &channels);
    image_result = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    kmeans(image_src, image_result, height, width, channels, 5);

    write_png(argv[2], image_result, height, width, channels);

    return 0;
}
