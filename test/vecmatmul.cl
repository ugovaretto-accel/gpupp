#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
typedef double real_t;
#else
typedef float real_t;
#endif

typedef unsigned uint;

#define COLUMN //2x speed increase

__kernel void VecMatMul( const __global real_t* M,
                         uint width,
                         uint height,
                         const __global real_t* V,
                         __global real_t* W ) 

{
 
#ifdef COLUMN // vector * matrix
  uint c = get_global_id( 0 );
  const __global real_t* column = M + c;
  real_t dp = 0.f;
  for( uint r = 0; r < height * width; r += width )
  {
    dp += column[ r ] * V[ c ];
  }
  W[ c ] = dp;
#else // matrix * vector
  uint r = get_global_id( 0 );
  const __global real_t* row = M + r * width;
  real_t dp = 0.f;
  for( uint c = 0; c != width; ++c )
  {
    dp += row[ c ] * V[ c ];
  }
  W[ r ] = dp;
 #endif 
}                         
                         
                                      
  
