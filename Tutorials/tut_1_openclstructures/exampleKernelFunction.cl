__kernel void exampleKernelFunction(global int* inputdata, global int* outputdata){

     int loopindex = get_global_id(0);
     int offset = 1;
     outputdata[loopindex] = inputdata[loopindex] - offset;

}