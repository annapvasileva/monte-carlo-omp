#include <omp.h>
#include <stdint.h>

#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <fstream>
#include <thread>

#include "hit.h"
#include "include/pcg_random.hpp"

struct Args {
    bool no_omp = false;
    int32_t num_of_threads = omp_get_max_threads();
    int64_t chunk_size = 1000;
    std::string input;
    std::string output;

    Args() = default;

    void ParseArgs(int32_t argc, char** argv) {
        for (int32_t i = 1; i < argc;) {
            std::string cur_arg = argv[i];
            if (cur_arg == "--no-omp") {
                no_omp = true;
                ++i;
            } else if (cur_arg == "--omp-threads") {
                if (i == argc - 1) {
                    std::cerr << "Not enough arguments given to --omp-threads: one argument is needed\n";
                    exit(1);
                }
                cur_arg = argv[i + 1];
                if (cur_arg != "default") {
                    num_of_threads = std::stoi(argv[i + 1]);
                }
                i += 2;
            } else if (cur_arg == "--input") {
                if (i == argc - 1) {
                    std::cerr << "Not enough arguments given to --input: one argument is needed\n";
                    exit(1);
                }
                input = argv[i + 1];
                i += 2;
            } else if (cur_arg == "--output") {
                if (i == argc - 1) {
                    std::cerr << "Not enough arguments given to --output: one argument is needed\n";
                    exit(1);
                }
                output = argv[i + 1];
                i += 2;
            } else if (cur_arg == "--chunk-size") {
                if (i == argc - 1) {
                    std::cerr << "Not enough arguments given to --chunk-size: one argument is needed\n";
                    exit(1);
                }
                chunk_size = std::stoll(argv[i + 1]);
                i += 2;
            } else {
                std::cerr << "Unknown argument: \"" << argv[i] << "\"\n";
                exit(1);
            }
        }
    }
};

void SerialExecution(int64_t num_of_samples, const std::string& output) {
    int32_t hits = 0;
    const float* limits = get_axis_range();

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    pcg32_fast generator(seed);
    std::uniform_real_distribution<float> x_distr(limits[0], limits[1]);
    std::uniform_real_distribution<float> y_distr(limits[2], limits[3]);
    std::uniform_real_distribution<float> z_distr(limits[4], limits[5]);
    
    for (int64_t i = 0; i < num_of_samples; ++i) {
        float x = x_distr(generator);
        float y = y_distr(generator);
        float z = z_distr(generator);
        if (hit_test(x, y, z)) {
            ++hits;
        }
    }
    
    float cube_volume = (limits[1] - limits[0]) * (limits[3] - limits[2]) * (limits[5] - limits[4]);
    
    FILE* out;
    out = fopen(output.c_str(), "w");
    fprintf(out, "%g\n", cube_volume * float(hits) / float(num_of_samples));
    fclose(out);
}

void ParallelExecution(int32_t num_of_threads, int64_t num_of_samples, const std::string& output, int64_t chunk_size) {
    int32_t hits = 0;
    const float* limits = get_axis_range();

    omp_set_dynamic(0);
    omp_set_num_threads(num_of_threads);
    #pragma omp parallel num_threads(num_of_threads)
    {
        int32_t cur_thread_hits = 0;

        std::random_device rd;
        pcg32_fast generator(rd());
        std::uniform_real_distribution<float> x_distr(limits[0], limits[1]);
        std::uniform_real_distribution<float> y_distr(limits[2], limits[3]);
        std::uniform_real_distribution<float> z_distr(limits[4], limits[5]);

    
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_of_samples; ++i) {
            float x = x_distr(generator);
            float y = y_distr(generator);
            float z = z_distr(generator);
            if (hit_test(x, y, z)) {
                ++cur_thread_hits;
            }
        }
        #pragma omp atomic
        hits += cur_thread_hits;
    }
    
    float cube_volume = (limits[1] - limits[0]) * (limits[3] - limits[2]) * (limits[5] - limits[4]);
    
    FILE* out;
    out = fopen(output.c_str(), "w");
    fprintf(out, "%g\n", cube_volume * float(hits) / float(num_of_samples));
    fclose(out);
}

int32_t main(int32_t argc, char** argv) {
    Args args;
    args.ParseArgs(argc, argv);

    std::ifstream in;
    in.open(args.input);

    if (!in.good()) {
        std::cerr << "Failed to open input file: " << args.input << "\n";
        exit(1);
    }

    std::string str_num_of_samples;
    in >> str_num_of_samples;
    int64_t num_of_samples = std::stoll(str_num_of_samples);

    double start = omp_get_wtime();
    if (args.no_omp) {
        SerialExecution(num_of_samples, args.output);
        args.num_of_threads = 0;
    } else {
        ParallelExecution(args.num_of_threads, num_of_samples, args.output, args.chunk_size);
    }
    double end = omp_get_wtime();

    printf("Time (%i thread(s)): %g ms\n", args.num_of_threads, (end - start) * 1000);
}