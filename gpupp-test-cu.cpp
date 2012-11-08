//
// Copyright (c) 2010 - Ugo Varetto
//
// This source code is free; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This source code is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this source code; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
// MA  02110-1301, USA.
// 

#include <string>
#include <vector>
#include "cuda/gpupp.h"
#include "utility/Timer.h"

// iota has been removed (why?) from STL long ago.
template< class FwdIt, class T > 
inline void iota( FwdIt begin, FwdIt end, T startVal )
{
    // compute increasing sequence into [begin, end)
    for (; begin != end; ++begin, ++startVal ) *begin = startVal;
}

typedef float real_t;

typedef std::vector< real_t > Array;

//------------------------------------------------------------------------------
/// Callback function object passed to scoped timer; will be invoked
/// with elapsed time upon timer destruction.
struct PrintTime
{
    void operator()( double t ) const
    {
        std::clog << "Time: " << t << " (ms)" << std::endl;
    }
};

//------------------------------------------------------------------------------
/// Callback function object passed to scoped timer; will be invoked
/// with elapsed time upon timer destruction. Specialization for GPU time
struct PrintTimeGPU
{
    void operator()( double t ) const
    {
        std::clog << "GPU Time: " << t << " (ms)" << std::endl;
    }
};

/// Shows how to use a high level C++ API to perform computation through OpenCL. 
void CUMatMulTest()
{
    typedef unsigned uint;

    static const std::string SEPARATOR =     
#ifdef WIN32
        "\\";
#else
        "/";
#endif
    std::string KERNEL_PATH;
    if( getenv( "CUDA_KERNEL_PATH" ) ) {
        KERNEL_PATH = std::string( getenv( "CUDA_KERNEL_PATH") ) +
                      SEPARATOR + 
                      std::string( "vecmatmul.ptx" );

    } else {
#ifdef WIN32
    KERNEL_PATH = "C:\\projects\\gpupp\\test\\vecmatmul.ptx";
#else
    KERNEL_PATH = "~/projects/gpupp/test/vecmatmul.ptx";
#endif
    }
    const std::string KERNEL_NAME( "VecMatMul" );
    const uint MATRIX_WIDTH = 1024; 
    const uint MATRIX_HEIGHT = MATRIX_WIDTH; // <- passed to OpenCL as uint
    const uint VECTOR_SIZE = MATRIX_WIDTH; // for M x V; MATRIX_HEIGHT for V x M
    const uint MATRIX_SIZE = MATRIX_WIDTH * MATRIX_HEIGHT;
    const uint MATRIX_BYTE_SIZE = sizeof( real_t ) * MATRIX_SIZE;
    const uint VECTOR_BYTE_SIZE = sizeof( real_t ) * VECTOR_SIZE;
    try
    {
        // (0) initialize
        //::cuInit(); // this sucks! it's required by CUDA, not clear why
                      // initialization cannot be performed on a per-context basis
                      // at context creation time;
                      // techiques like invoking cuInit() from a constructor won't
                      // work if the global object is defined inside a (dynamic) library;
                      // this is currently addressed by invoking the init function from
                      // within the context creation function
        // (1) init data
        Array inMatrix( MATRIX_SIZE, real_t( 0 ) );
        Array inVector( VECTOR_SIZE, real_t( 0 ) );
        Array outVector( inVector );
        iota( inMatrix.begin(), inMatrix.end(), real_t( 0 ) );
        iota( inVector.begin(), inVector.end(), real_t( 0 ) );
        // (2) create kernel
        std::string buildOutput;  // compiler output
        std::string buildOptions; // e.g. -DDOUBLE
        // NOT POSSIBLE WITH CUDA SINCE KERNEL ARE *ALWAYS* PRECOMPILED
        // const bool TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE = true; 
        CUDAExecutionContext ec = 
            CreateContextAndKernelFromFile( 0, //<- device number; use first available
                                            KERNEL_PATH,
                                            KERNEL_NAME );
                                            //NO RUN-TIME BUILD AVAILABLE WITH CUDA
                                            //buildOutput,
                                            //buildOptions );
                                            //TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE );
        // OpenCL only
        //std::clog << buildOutput << std::endl;
        //if( ec.wgroupSize > 0 ) std::clog << "Computed optimal workgroup size: " << ec.wgroupSize << std::endl;
        //else std::clog << "Could not compute optimal workgroup size"  << std::endl;
        // (3) allocate input and otput buffer that will be passed to kernel function
        CUMemObj  inMatD( ec.context, MATRIX_BYTE_SIZE );
        CUMemObj  inVecD( ec.context, VECTOR_BYTE_SIZE );
        CUMemObj outVecD( ec.context, VECTOR_BYTE_SIZE );
        // (4) copy data into input buffers
        CUDACopyHtoD( inMatD, &inMatrix[ 0 ] );
        CUDACopyHtoD( inVecD, &inVector[ 0 ] );
        // (5) execute kernel
        SizeArray globalWGroupSize( 1, MATRIX_HEIGHT ); 
        SizeArray  localWGroupSize( 1, 256  );
        // kernel signature:
        // void VecMatMul( const real_t* M,
        //                 uint width,
        //                 uint height,
        //                 const real_t* V,
        //                 real_t* W )
        {
			ScopedCBackTimer< PrintTimeGPU, CUDATimer > ptGPU;
            //ScopedCBackTimer< PrintTime > pt;
            InvokeKernelSync( ec, globalWGroupSize, localWGroupSize,
                                ( VArgList(),  //<- Marks the beginning of a variable argument list
                                  inMatD,
                                  MATRIX_WIDTH,
                                  MATRIX_HEIGHT,
                                  inVecD,
                                  outVecD
                                )              //<- Marks the end of a variable argument list
                             );
        }
        // (6) read back results
        CUDACopyDtoH( outVecD, &outVector[ 0 ] );
        // print first two and last elements
        std::cout << "vector[0]    = " << outVector[ 0 ] << '\n';
        std::cout << "vector[1]    = " << outVector[ 1 ] << '\n';
        std::cout << "vector[last] = " << outVector.back() << std::endl;
        // (7) release resources: done automatically by resouce wrapper destructor
    }
    catch( const std::exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
}

//------------------------------------------------------------------------------
void ListPlatforms()
{
#if 1
     PrintDevicesInfo( std::cout, QueryDevicesInfo() ); 
#endif
}


//------------------------------------------------------------------------------
int main( int argc, char** argv )
{    
    ListPlatforms();
    CUMatMulTest();
#ifdef _MSC_VER
#ifdef _DEBUG
    std::cout << "\n<press Enter to exit>" << std::endl;
    ::getchar();
#endif
#endif
    return 0;
}
