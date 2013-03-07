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
#include "opencl/gpupp.h"
#include "utility/Timer.h"

// iota was removed (why?) from STL long ago.
template< class FwdIt, class T > 
inline void iota( FwdIt begin, FwdIt end, T startVal ) {
    // compute increasing sequence into [begin, end)
    for(; begin != end; ++begin, ++startVal ) *begin = startVal;
}

typedef float real_t;

typedef std::vector< real_t > Array;

//------------------------------------------------------------------------------
/// Callback function object passed to scoped timer; will be invoked
/// with elapsed time upon timer destruction.
struct PrintTime {
    void operator()( double t ) const {
        std::clog << "Time: " << t << " (ms)" << std::endl;
    }
};

/// Shows how to use a high level C++ API to perform computation through OpenCL. 
void CLMatMulTest( const char* platformName, int deviceNum, int matrixSize ) {
    typedef unsigned uint;

    static const std::string SEPARATOR =
#ifdef WIN32
        "\\";
#else
        "/";
#endif
    std::string KERNEL_PATH;
    if( getenv( "OPENCL_KERNEL_PATH" ) ) {
        KERNEL_PATH = std::string( getenv( "OPENCL_KERNEL_PATH") ) +
                      SEPARATOR +
                      std::string( "matmul.cl" );

    } else {
#ifdef WIN32
        KERNEL_PATH = "C:\\projects\\gpupp\\test\\matmul.cl";
#else
        KERNEL_PATH = "/project/csstaff/uvaretto/src/gpupp/test/matmul.cl";
#endif
        std::cout << "OpenCL default kernel path: " << KERNEL_PATH << std::endl;
        std::cout << "Set the default OpenCL kernel path "
                     "with the OPENCL_KERNEL_PATH env var" << std::endl;    
    }
    const std::string KERNEL_NAME( "MatMul" );
    const uint MATRIX_WIDTH = matrixSize; // <- passed to OpenCL as uint
    const uint MATRIX_HEIGHT = MATRIX_WIDTH; // <- passed to OpenCL as uint
    const size_t MATRIX_SIZE = MATRIX_WIDTH * MATRIX_HEIGHT;
    const size_t MATRIX_BYTE_SIZE = sizeof( real_t ) * MATRIX_SIZE;
    try {

        // (1) init data
        Array A( MATRIX_SIZE, real_t( 0 ) );
        Array B( MATRIX_SIZE, real_t( 0 ) );
        Array C( MATRIX_SIZE, real_t( 0 ) );
        iota( A.begin(), A.end(), real_t( 0 ) );
        iota( B.begin(), B.end(), real_t( -MATRIX_SIZE ) );
        // (2) create kernel
        std::string buildOutput;  // compiler output
        std::string buildOptions; // e.g. -DDOUBLE
        const bool TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE = true; 
        CLExecutionContext ec = 
            CreateContextAndKernelFromFile( platformName, //<- platform name
                                            CL_DEVICE_TYPE_ALL, //<- select all devices available on platform
                                            deviceNum, //<- device number; use first available
                                            KERNEL_PATH, //<- full path to file containing kernel source code
                                            KERNEL_NAME, //<- name of kernel function
                                            buildOutput, //<- compiler output
                                            buildOptions, //<- compiler options
                                            TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE,
                                            CL_QUEUE_PROFILING_ENABLE );
        std::clog << buildOutput << std::endl;
        if( ec.wgroupSize > 0 ) std::clog << "Computed optimal workgroup size: " 
                                          << ec.wgroupSize << std::endl;
        else std::clog << "Could not compute optimal workgroup size"  << std::endl;
        // (2.1) enable command queue profiling
        // DEPRECATED IN OpenCL 1.1 SINCE NOT THREAD SAFE 
		//EnableProfiling( ec.commandQueue );

        // (3) allocate input and otput buffer that will be passed
        // to kernel function
        CLMemObj  dA( ec.context, MATRIX_BYTE_SIZE, CL_MEM_READ_ONLY );
        CLMemObj  dB( ec.context, MATRIX_BYTE_SIZE, CL_MEM_READ_ONLY );
        CLMemObj  dC( ec.context, MATRIX_BYTE_SIZE, CL_MEM_WRITE_ONLY );

        // (4) copy data into input buffers
        CLCopyHtoD( ec.commandQueue, dA, &A[ 0 ] );
        CLCopyHtoD( ec.commandQueue, dB, &B[ 0 ] );
        // (5) execute kernel
        SizeArray globalWGroupSize( 1, MATRIX_HEIGHT ); 
        SizeArray  localWGroupSize( 1, ec.wgroupSize > 0 ? ec.wgroupSize : 256  );
        cl_event kernelEvent = cl_event();
        // kernel signature:
        // void MatMul( const __global real_t* restrict A,
        //              const __global real_t* restrict B, 
        //              __global real_t* restrict C, 
        //              uint width,
        //              uint height )
        {
            ScopedCBackTimer< PrintTime > pt;
            kernelEvent = InvokeKernelSync( ec, globalWGroupSize, localWGroupSize,
                                ( VArgList(),  //<- Marks the beginning of a variable argument list
                                  cl_mem( dA ),
                                  cl_mem( dB )
                                  cl_mem( dC ),
                                  MATRIX_WIDTH,
                                  MATRIX_HEIGHT )
                                )              //<- Marks the end of a variable argument list
                             );
        }
        // (6) read back results
        CLCopyDtoH( ec.commandQueue, dC, &C[ 0 ] );
        // print first two and last elements
        std::cout << "vector[0]    = " << C[ 0 ] << '\n';
        std::cout << "vector[1]    = " << C[ 1 ] << '\n';
        std::cout << "vector[last] = " << C.back() << std::endl;
        // (6.1) print profilng information
        std::cout << "Kernel execution latency (ms): " 
                  << ProfilingInfo( kernelEvent ).Latency()       << std::endl;
        std::cout << "Kernel execution time (ms):    " 
                  << ProfilingInfo( kernelEvent ).ExecutionTime() << std::endl;
        // (7) release resources
        //ReleaseExecutionContext( ec );
    }
    catch( const std::exception& e ) {
        std::cerr << e.what() << std::endl;
    }
}

//------------------------------------------------------------------------------
void ListPlatforms() {
    Platforms platforms = QueryPlatforms();
    PrintPlatformsInfo( std::cout, platforms );
}


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {    
    ListPlatforms();
    if( argc < 2 ) {
        std::cout << "usage: " << argv[0] 
                  << " <platform name e.g. NVIDIA CUDA> "
                     "[device id - default is 0] "
                     "[matrix size - default is 1024"
                  << std::endl;
        return 0;          
    }
    int deviceNum = 0;
    if( argc > 2 ) deviceNum = atoi( argv[2] );
    int matrixSize = 1024;
    if( argc > 3 ) matrixSize = atoi( argv[3] );
    CLMatMulTest( argv[1], deviceNum, matrixSize );
#ifdef _MSC_VER
#ifdef _DEBUG
    std::cout << "\n<press Enter to exit>" << std::endl;
    ::getchar();
#endif
#endif
    return 0;
}
