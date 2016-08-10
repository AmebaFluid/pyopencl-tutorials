// Each workgroup processes a part of the scalar product. Their values are partial results.
// A final reduction step over all partial results is performed to obtain the result.

__kernel void scalarproduct(__global float* vectorA, __global float* vectorB, __local float* partial_result, __local float* product){

    // number of workitems in the workgroup for a given dimension:
    int group_size = get_local_size(0);

    product[]
    int global_index = get_global_id(0);


    result = vectorA[index] * vectorB[index];
}