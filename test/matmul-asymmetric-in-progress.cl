#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif

//USE 8x8 on XEON PHI IF NOT IT CRASHES
//pass it as a build option
//#define TILE_SIZE 8 

// return matrix element given block and indices of element in block
real_t get_matrix_element( __global const real_t* restrict m, //matrix
                           int blockCol,    //column index of output block 
                           int blockRow,    //row index of output row
                           int col,         //local column index of block element
                           int row,         //local row index of block element 
                           int num_columns) {                                           
  
    return m[ ( blockRow * get_local_size(1) + row ) 
              * num_columns + blockCol * get_local_size(0) + col ];

}


#if 1
__kernel void
MatMul( __global const real_t* restrict A, 
        __global const real_t* restrict B,
        __global real_t* restrict C,
        int width,
        int height ) {

    __local real_t M1[TILE_HEIGHT][TILE_WIDTH];
    __local real_t M2[TILE_WIDTH][TILE_WIDTH];

    const int blockRow = get_group_id(1); 
    const int blockCol = get_group_id(0);
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    real_t out = 0;
    for( int b = 0; b < width / get_local_size(0); ++b ) {
        //copy data into shared memory
        M1[ row ][ col ] = get_matrix_element( A, b, blockRow, col, row, width );
        M2[ col ][ row ] = get_matrix_element( B, blockCol, b, row, col, width );
        barrier(CLK_LOCAL_MEM_FENCE); // required to guarantee that data are computed before next step
                                      // where a thread accesses data computed by other threads
        for( int c = 0; c != get_local_size(0); ++c ) {
            out += M1[ row ][ c ] * M2[ c ][ col ];           
        }
        barrier(CLK_LOCAL_MEM_FENCE); // required to avoid that some threads start modifying
                         // data in cache before all threads have exited for loop    
    }
    const int idx = ( blockRow * get_local_size(1) + row ) * width + blockCol * get_local_size(0) + col;
    C[ idx ] = out;     

}
#else
__kernel void
MatMul( __global const real_t* A, 
        __global const real_t* B,
        __global real_t* C,
        int width,
        int height ) {

    int row = get_global_id( 1 );
    int col = get_global_id( 0 );
    real_t out = 0;
    for( int b = 0; b != width ; ++b ) {
        out += A[ row * width + b ] * B[ b * width + col ];     
    }
    C[ row * width + col ] = out;     
}
#endif                         
                                      
  
