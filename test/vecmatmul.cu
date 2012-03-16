#ifdef DOUBLE__
typedef double real_t;
#else
typedef float real_t;
#endif

//#define COLUMN //2x speed increase!

typedef unsigned uint;
extern "C" __global__ void VecMatMul( const real_t* M,
									  uint width,
									  uint height,
									  const real_t* V,
									  real_t* W )
{
 
#ifdef COLUMN // vector * matrix
  uint c = blockIdx.x * blockDim.x + threadIdx.x;
  //if( c >= height ) return;
  const real_t* column = M + c;
  real_t dp = 0.f;
  for( uint r = 0; r < height * width; r += width )
  {
    dp += column[ r ] * V[ c ];
  }
  W[ c ] = dp;
#else // matrix * vector
  uint r = blockIdx.x * blockDim.x + threadIdx.x;
  //if( r >= width ) return;
  const real_t* row = M + r * width;
  real_t dp = 0.f;
  for( uint c = 0; c != width; ++c )
  {
    dp += row[ c ] * V[ c ];
  }
  W[ r ] = dp;
 #endif 
}                         
                         
                                      
  