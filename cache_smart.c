#include <stdlib.h>
#include <stdio.h>
#include "timer.h"

#define CACHELINE 64
#define L1 32768
#define L2 262144
#define L3 6291456

void reset_vector(double* vals, int vector_size)
{
    for (int i = 0; i < vector_size; i++)
    {
        vals[i] = 1.0;
    }
}

double read_vector(double* vals, int vector_size, int n_iter)
{
    double val;
    double sum = 0;
    for (int iter = 0; iter < n_iter; iter++)
    {
        val = 1.0 / (iter+1);
        for (int i = 0; i < vector_size; i++)
        {
            sum += vals[i] * val; 
        }
    }
    return sum;
}

double read_vector_skip_cacheline(double* vals, int vector_size, int n_iter, int cacheline_dbl)
{
    double val;
    double sum = 0;
    for (int iter = 0; iter < n_iter; iter++)
    {
        val = 1.0 / (iter+1);
        for (int i = 0; i < cacheline_dbl; i++)
        {
            for (int j = 0; j < vector_size; j+= cacheline_dbl)
            {
                sum += vals[j+i] * val;
            }
        }
    }
    return sum;
}

void write_vector(double* vals, int vector_size, int n_iter)
{
    double val;
    double alpha = 0;
    for (int iter = 0; iter < n_iter; iter++)
    {
        val = 1.0 / (iter+1);
        for (int i = 0; i < vector_size; i++)
        {
            vals[i] = vals[i]*alpha + val; 
        }
    }
}

void write_vector_skip_cacheline(double* vals, int vector_size, int n_iter, int cacheline_dbl)
{
    double val;
    double alpha = 0;
    for (int iter = 0; iter < n_iter; iter++)
    {
        val = 1.0 / (iter+1);
        for (int i = 0; i < cacheline_dbl; i++)
        {
            for (int j = 0; j < vector_size; j+= cacheline_dbl)
            {
                vals[j+i] = vals[j+1]*alpha + val;
            }
        }
    }
}

double norm(int vector_size, double* vals)
{
    double sum = 0;
    for (int i = 0; i < vector_size; i++)
        sum += (vals[i]*vals[i]);
    return sum;
}



void print_data(int n_access, double start, double end, double result)
{
    double seconds = get_seconds(start, end);
    long bytes = (n_access*sizeof(double));
    double rate = get_rate(seconds, bytes);
    double grate = get_grate(rate);
    printf("Result %e, Seconds %e, Seconds Per Double %e, Gbytes/sec %e\n", result, seconds, seconds / n_access, grate);
}

int main(int argc, char* argv[])
{
    int cacheline_dbl = CACHELINE/sizeof(double);

    int read = 1 - atoi(argv[1]);

    int n_access = 1000000000;
    int vector_size = (2*L3)/sizeof(double);
    int n_iter = n_access / vector_size;
    double* vals = (double*)malloc(vector_size*sizeof(double));
    double start, end;
    double result;


    /****************************************
     **** Main Memory Costs
     ***************************************/
    if (read) printf("Reading From Main Memory...\n");
    else printf("Writing To Main Memory...\n");

    // Time Reading from Main Memory, striding cacheline
    printf("1. Striding Cacheline:\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
    }
    else
    {
        write_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);

    // Time Reading from Main Memory, with cacheline benefit
    printf("2. Utilizing Cacheline\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector(vals, vector_size, n_iter);
    }
    else
    {
        write_vector(vals, vector_size, n_iter);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);
    printf("\n\n");




    /****************************************
     **** L3 Cache Costs
     ***************************************/
    if (read) printf("Reading from L3 cache...\n");
    else printf("Writing to L3 cache...\n");
    vector_size = (2*L2)/sizeof(double);
    n_iter = n_access / vector_size;

    // Time Reading from L3 Cache. striding cacheline
    printf("1. Striding Cacheline:\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
    }
    else
    {
        write_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);

    // Time Reading from L3 Cache, with cacheline benefit
    printf("2. Utilizing Cacheline\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector(vals, vector_size, n_iter);
    }
    else
    {
        write_vector(vals, vector_size, n_iter);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);
    printf("\n\n");





    /****************************************
     **** L2 Cache Costs
     ***************************************/
    if (read) printf("Reading from L2 cache...\n");
    else printf("Writing to L2 cache...\n");
    vector_size = (2*L1) / sizeof(double);
    n_iter = n_access / vector_size;

    // Time Reading from L2 Cache, striding cacheline
    printf("1. Striding Cacheline:\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
    }
    else
    {
        write_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);

    // Time Reading from L3 Cache, with cacheline benefit
    printf("2. Utilizing Cacheline\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector(vals, vector_size, n_iter);
    }
    else
    {
        write_vector(vals, vector_size, n_iter);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);
    printf("\n\n");




    /****************************************
     **** L1 Cache Costs
     ***************************************/
    if (read) printf("Reading from L1 cache...\n");
    else printf("Writing to L1 cache...\n");
    vector_size = (L1/2) / sizeof(double);
    n_iter = n_access / vector_size;


    // Time Reading from L1 Cache, striding cacheline
    printf("1. Striding Cacheline:\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
    }
    else
    {
        write_vector_skip_cacheline(vals, vector_size, n_iter, cacheline_dbl);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);

    // Time Reading from L1 Cache, with cacheline benefit
    printf("2. Utilizing Cacheline\n");
    reset_vector(vals, vector_size);
    start = get_time();
    if (read)
    {
        result = read_vector(vals, vector_size, n_iter);
    }
    else
    {
        write_vector(vals, vector_size, n_iter);
        result = norm(vector_size, vals);
    }
    end = get_time();
    print_data(n_access, start, end, result);
    printf("\n\n");


    free(vals);

    return 0;
}










