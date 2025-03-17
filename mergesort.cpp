#include <iostream>
#include <pthread.h>
#include <vector>

using namespace std;

struct ThreadArgs {
    double* array;
    uint64_t left;
    uint64_t right;
    uint64_t depth;
    uint64_t maxDepth;
};

pthread_mutex_t mutexPart;
uint64_t currentPart = 0;

void merge(double* arr, uint64_t l, uint64_t m, uint64_t r) {
    uint64_t i, j, k;
    uint64_t n1 = m - l + 1;
    uint64_t n2 = r - m;

    // Create temp arrays
    double* L = new double[n1];
    double* R = new double[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp arrays back into arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

void* mergeSort(void* args) {
    ThreadArgs* arg = (ThreadArgs*)args;
    if (arg->left < arg->right) {
        if (arg->depth >= arg->maxDepth) {
            // Perform normal mergesort if max depth reached
            uint64_t mid = arg->left + (arg->right - arg->left) / 2;
            ThreadArgs leftArgs = {arg->array, arg->left, mid, arg->depth + 1, arg->maxDepth};
            ThreadArgs rightArgs = {arg->array, mid + 1, arg->right, arg->depth + 1, arg->maxDepth};
            mergeSort(&leftArgs);
            mergeSort(&rightArgs);
            merge(arg->array, arg->left, mid, arg->right);
        } else {
            uint64_t mid = arg->left + (arg->right - arg->left) / 2;

            pthread_t leftThread, rightThread;
            ThreadArgs leftArgs = {arg->array, arg->left, mid, arg->depth + 1, arg->maxDepth};
            ThreadArgs rightArgs = {arg->array, mid + 1, arg->right, arg->depth + 1, arg->maxDepth};

            pthread_create(&leftThread, NULL, mergeSort, &leftArgs);
            pthread_create(&rightThread, NULL, mergeSort, &rightArgs);

            pthread_join(leftThread, NULL);
            pthread_join(rightThread, NULL);

            merge(arg->array, arg->left, mid, arg->right);
        }
    }
    return NULL;
}

void cpu_merge(double *unSorted, uint64_t sizeOfArray, int threadNum) {

    // Calculate max depth based on number of threads
    uint64_t maxDepth = 0;
    while ((1 << maxDepth) < threadNum) maxDepth++;

    pthread_mutex_init(&mutexPart, NULL);

    ThreadArgs args = {unSorted, 0, sizeOfArray - 1, 0, maxDepth};
    mergeSort(&args);


    pthread_mutex_destroy(&mutexPart);

}
