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

/// Shows how to use a high level C++ API to perform computation through OpenCL. 
void CLMatMulTest()
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
        KERNEL_PATH = std::string( getenv( "OPENCL_KERNEL_PATH") ) +
                      SEPARATOR +
                      std::string( "vecmatmul.cl" );

    } else {
#ifdef WIN32
    KERNEL_PATH = "C:\\projects\\gpupp\\test\\vecmatmul.ptx";
#else
    KERNEL_PATH = "~/projects/gpupp/test/vecmatmul.ptx";
#endif
    }

    const std::string KERNEL_NAME( "VecMatMul" );
    const uint MATRIX_WIDTH = 1024; // <- passed to OpenCL as uint
    const uint MATRIX_HEIGHT = MATRIX_WIDTH; // <- passed to OpenCL as uint
    const size_t VECTOR_SIZE = MATRIX_WIDTH; // for M x V; MATRIX_HEIGHT for V x M
    const size_t MATRIX_SIZE = MATRIX_WIDTH * MATRIX_HEIGHT;
    const size_t MATRIX_BYTE_SIZE = sizeof( real_t ) * MATRIX_SIZE;
    const size_t VECTOR_BYTE_SIZE = sizeof( real_t ) * VECTOR_SIZE;
    try
    {

        // (1) init data
        Array inMatrix( MATRIX_SIZE, real_t( 0 ) );
        Array inVector( VECTOR_SIZE, real_t( 0 ) );
        Array outVector( inVector );
        iota( inMatrix.begin(), inMatrix.end(), real_t( 0 ) );
        iota( inVector.begin(), inVector.end(), real_t( 0 ) );
        // (2) create kernel
        std::string buildOutput;  // compiler output
        std::string buildOptions; // e.g. -DDOUBLE
        const bool TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE = true; 
        CLExecutionContext ec = 
            CreateContextAndKernelFromFile( "NVIDIA CUDA", //<- platform name
                                            CL_DEVICE_TYPE_ALL, //<- select all devices available on platform
                                            0, //<- device number; use first available
                                            KERNEL_PATH, //<- full path to file containing kernel source code
                                            KERNEL_NAME, //<- name of kernel function
                                            buildOutput, //<- compiler output
                                            buildOptions, //<- compiler options
                                            TRY_TO_COMPUTE_OPTIMAL_WGROUP_SIZE,
                                            CL_QUEUE_PROFILING_ENABLE );
        std::clog << buildOutput << std::endl;
        if( ec.wgroupSize > 0 ) std::clog << "Computed optimal workgroup size: " << ec.wgroupSize << std::endl;
        else std::clog << "Could not compute optimal workgroup size"  << std::endl;
        // (2.1) enable command queue profiling
        // DEPRECATED IN OpenCL 1.1 SINCE NOT THREAD SAFE 
		//EnableProfiling( ec.commandQueue );

        // (3) allocate input and otput buffer that will be passed
        // to kernel function
        CLMemObj  inMatD( ec.context, MATRIX_BYTE_SIZE, CL_MEM_READ_ONLY );
        CLMemObj  inVecD( ec.context, VECTOR_BYTE_SIZE, CL_MEM_READ_ONLY );
        CLMemObj outVecD( ec.context, VECTOR_BYTE_SIZE, CL_MEM_WRITE_ONLY );
        // (4) copy data into input buffers
        CLCopyHtoD( ec.commandQueue, inMatD, &inMatrix[ 0 ] );
        CLCopyHtoD( ec.commandQueue, inVecD, &inVector[ 0 ] );
        // (5) execute kernel
        SizeArray globalWGroupSize( 1, MATRIX_HEIGHT ); 
        SizeArray  localWGroupSize( 1, ec.wgroupSize > 0 ? ec.wgroupSize : 256  );
        cl_event kernelEvent = cl_event();
        // kernel signature:
        // void VecMatMul( const __global real_t* M,
        //                 uint width,
        //                 uint height,
        //                 const __global real_t* V,
        //                 __global real_t* W )
        {
            ScopedCBackTimer< PrintTime > pt;
            kernelEvent = InvokeKernelSync( ec, globalWGroupSize, localWGroupSize,
                                ( VArgList(),  //<- Marks the beginning of a variable argument list
                                  inMatD.GetCLMemHandle(),
                                  MATRIX_WIDTH ,
                                  MATRIX_HEIGHT ,
                                  cl_mem( inVecD ) ,
                                  cl_mem( outVecD )
                                )              //<- Marks the end of a variable argument list
                             );
        }
        // (6) read back results
        CLCopyDtoH( ec.commandQueue, outVecD, &outVector[ 0 ] );
        // print first two and last elements
        std::cout << "vector[0]    = " << outVector[ 0 ] << '\n';
        std::cout << "vector[1]    = " << outVector[ 1 ] << '\n';
        std::cout << "vector[last] = " << outVector.back() << std::endl;
        // (6.1) print profilng information
        std::cout << "Kernel execution latency (ms): " << ProfilingInfo( kernelEvent ).Latency()       << std::endl;
        std::cout << "Kernel execution time (ms):    " << ProfilingInfo( kernelEvent ).ExecutionTime() << std::endl;
        // (7) release resources
        //ReleaseExecutionContext( ec );
    }
    catch( const std::exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
}

//------------------------------------------------------------------------------
void ListPlatforms()
{
    Platforms platforms = QueryPlatforms();
    PrintPlatformsInfo( std::cout, platforms );
}


//------------------------------------------------------------------------------
int main( int argc, char** argv )
{    
    ListPlatforms();
    CLMatMulTest();
#ifdef _MSC_VER
#ifdef _DEBUG
    std::cout << "\n<press Enter to exit>" << std::endl;
    ::getchar();
#endif
#endif
    return 0;
}
