///\file opencl/gpupp.h OpenCL funtions and types

#ifndef GPUPP_H_
#define GPUPP_H_
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

#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>
#include <map>
#include <CL/cl.h>
#include "../utility/varargs.h"
#include "../utility/ResourceHandler.h"

///Context resource name
struct ContextName
{
    operator const char*() const { return "Context"; }
};
///Kernel resource name
struct KernelName
{
    operator const char*() const { return "Kernel"; }
};
///Program resource name
struct ProgramName
{
    operator const char*() const { return "Program"; }
};
///Command queue resource name
struct CommandQueueName
{
    operator const char*() const { return "CommandQueue"; }
};

///Context resource handler
typedef ResourceHandler< cl_context,
                         cl_int,
                         ::clRetainContext,
                         ::clReleaseContext,
                         ContextName,
                         CL_SUCCESS > HContext;
///Kernel resource handler
typedef ResourceHandler< cl_kernel,
                         cl_int,
                         ::clRetainKernel,
                         ::clReleaseKernel,
                         KernelName,
                         CL_SUCCESS > HKernel;
///Program resource handler
typedef ResourceHandler< cl_program,
                         cl_int,
                         ::clRetainProgram,
                         ::clReleaseProgram,
                         ProgramName,
                         CL_SUCCESS > HProgram; 
///Command queue resource handler
typedef ResourceHandler< cl_command_queue,
                         cl_int,
                         ::clRetainCommandQueue,
                         ::clReleaseCommandQueue,
                         CommandQueueName,
                         CL_SUCCESS > HCommandQueue; 

//-----------------------------------------------------------------------------
/// Execution context with complete information on execution environment.
/// This class groups together in a bundle context, command queue, and kernel.
/// Automatic resource management is performed by the data members.
struct CLExecutionContext
{
    /// OpenCL platform id
    cl_platform_id platform;
    /// OpenCL computing device id
    cl_device_id device;
    /// OpenCL context
    HContext context;
    //cl_context context;
    /// OpenCL command queue: note that it is possible to have
    /// more than a command queue per context so it is valid to have
    /// multiple instances of this class with same context and different
    /// command queues
    HCommandQueue commandQueue;
    //cl_command_queue commandQueue;
    /// OpenCL compiled program
    HProgram program;
    //cl_program program;
    /// OpenCL kernel function
    HKernel kernel;
    //cl_kernel kernel;
    /// "Optimal" workgroup size as returned by run-time 
    /// \note some environments do not support automatic workgroup size computation
    size_t wgroupSize;
    /// Amount of memory in bytes used by the kernel as returned by run-time
    size_t localMemSize; //local memory size as returned by run-time
    /// Default constructor: only initializes meaningful memebers used to detect
    /// proper initialization of class instances
    CLExecutionContext() : platform( cl_platform_id() ), device( cl_device_id() ), wgroupSize( 0 ), localMemSize( 0 ) {}
    /// Constructor as used in CreateCLContext function.
    CLExecutionContext( cl_platform_id pl,
                        cl_device_id d,
                        const HContext& ctx ) : 
                        platform( pl ), device( d ),
                            context( ctx ), wgroupSize( 0 ), localMemSize( 0 ) {}
    
};

//------------------------------------------------------------------------------
/// Create OpenCL Context.
/// \param[in] platformString platform identifier e.g. "NVIDIA CUDA" or "ATI Stream"
/// \param[in] deviceNum index of device to use
/// \param[in] deviceType type of device requested
/// \return valid execution context
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CLExecutionContext CreateCLExecutionContext( const std::string& platformString,
                                             int deviceNum,
                                             cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT );


typedef std::map< std::string, std::string > DeviceInfoMap;
typedef std::vector< DeviceInfoMap > Devices;

//------------------------------------------------------------------------------
/// Platform information: name and number of available devices.
struct PlatformInfo
{
    std::string vendor;     //!< CL_PLATFORM_VENDOR
    std::string profile;    //!< CL_PLATFORM_PROFILE
    std::string version;    //!< CL_PLATFORM_VERSION
    std::string extensions; //!< CL_PLATFORM_EXTENSIONS
    std::string name;       //!< CL_PLATFORM_NAME
    ///Sequence of 'Device property name' -> 'Device property value' maps
    Devices devices;        
};

typedef std::vector< PlatformInfo > Platforms;

//------------------------------------------------------------------------------
/// Returns a sequence of records containing the platform names and number of
/// devices available on each platform.
Platforms QueryPlatforms();

//------------------------------------------------------------------------------
/// Print platform info.
void PrintPlatformInfo( std::ostream& os, const PlatformInfo& pi,
                        const std::string& deviceIndent = "\t" );

//------------------------------------------------------------------------------
/// Print platform info.
void PrintPlatformsInfo( std::ostream& os, const Platforms& p,
                         const std::string& deviceIndent = "\t" );

//-----------------------------------------------------------------------------
/// Create command queue inside valid execution context.
/// \attention it is the responsibility of the client code to release previously allocated queues
/// \param[in] ec valid execution context
/// \return copy of input context containing handle of allocated command queue
/// \throw std::logic_error in case passed execution context is invalid
/// \throw std::runtime_error in case of errors creating the command queue
CLExecutionContext CreateCommandQueue( CLExecutionContext ec );

//-----------------------------------------------------------------------------
/// Build kernel from source text. Valid program, kernel and info are added
/// into a copy of the passed context.
/// \param[in] ec valid execution context
/// \param[in] kernelSrc source code of program
/// \param[in] kernelName name of kernel function
/// \param[in] buildOptions build options passed to OpenCL compiler
/// \param[in] computeWGroupSize ask run-time to compute optimal workgroup size
/// \param[out] buildOutput log from compiler
/// \return copy of passed execution context with added program and kernel handles
/// \throw std::runtime_error in case of errors while invoking OpenCL functions
CLExecutionContext BuildKernel( CLExecutionContext ec,
                                const std::string& kernelSrc,
                                const std::string& kernelName,
                                std::string& buildOutput,
                                const std::string& buildOptions = "",
                                bool computeWGroupSize = false );

//-----------------------------------------------------------------------------
/// Wrapper function that simply calls CreateCLExecutionContext,
/// CreateCommandQueue and BuildKernel in sequence.
/// \param[in] platformString platform identifier e.g. "NVIDIA CUDA" or "ATI Stream"
/// \param[in] deviceType OpenCL device type e.g. \c CL_DEVICE_TYPE_GPU; valid values are:
/// - CL_DEVICE_TYPE_CPU
/// - CL_DEVICE_TYPE_GPU
/// - CL_DEVICE_TYPE_ACCELERATOR
/// - CL_DEVICE_TYPE_DEFAULT
/// - CL_DEVICE_TYPE_ALL
/// \param[in] deviceNum index of device of type \c deviceType to use
/// \param[in] kernelSrc source code of program
/// \param[in] kernelName name of kernel function
/// \param[in] buildOptions build options passed to OpenCL program compiler
/// \param[in] computeWGroupSize ask run-time to compute optimal workgroup size
/// \param[out] buildOutput output log from compiler
/// \return valid CLExecutionContext with all members initialized
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CLExecutionContext CreateContextAndKernel( const std::string& platformString,
                                           cl_device_type deviceType,
                                           int deviceNum,     
                                           const std::string& kernelSrc,
                                           const std::string& kernelName,
                                           std::string& buildOutput,
                                           const std::string& buildOptions = "",
                                           bool computeWGroupSize = false, 
										   cl_command_queue_properties prop = cl_command_queue_properties() );

//-----------------------------------------------------------------------------
/// Wrapper function that simply calls CreateContextAndKernel, which in turns
/// calls CreateCLExecutionContext, CreateCommandQueue and BuildKernel in sequence.
/// \param[in] platformString platform identifier e.g. "NVIDIA CUDA" or "ATI Stream"
/// \param[in] deviceType OpenCL device type e.g. \c CL_DEVICE_TYPE_GPU; valid values are:
/// - CL_DEVICE_TYPE_CPU
/// - CL_DEVICE_TYPE_GPU
/// - CL_DEVICE_TYPE_ACCELERATOR
/// - CL_DEVICE_TYPE_DEFAULT
/// - CL_DEVICE_TYPE_ALL
/// \param[in] deviceNum index of device of type \c deviceType to use
/// \param[in] kernelPath full path to file containing source code of OpenCL program
/// \param[in] kernelName name of kernel function
/// \param[in] buildOptions build options passed to OpenCL program compiler
/// \param[in] computeWGroupSize ask run-time to compute optimal workgroup size
/// \param[in] prop command queue properties
/// \param[out] buildOutput output log from compiler
/// \return valid CLExecutionContext with all members initialized
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CLExecutionContext CreateContextAndKernelFromFile( const std::string& platformString,
                                                   cl_device_type deviceType,
                                                   int deviceNum,
                                                   const std::string& kernelPath,
                                                   const std::string& kernelName,
                                                   std::string& buildOutput,
                                                   const std::string& buildOptions = "",
                                                   bool computeWGroupSize = false,
												   cl_command_queue_properties prop = cl_command_queue_properties() );

//------------------------------------------------------------------------------
/// Wrapper for OpenCL memory object which performs automatic resource
/// deallocation and reference counting.
class CLMemObj
{
    CLMemObj(); // cannot default construct since it requires a valid context;
                // we could enable this by adding SetContext() method; this
                // would however make things messy
public:
    CLMemObj( cl_context ctx,
              size_t size,    
              cl_mem_flags flags = CL_MEM_READ_WRITE,
              void* hostPtr = 0 ) 
              : ctx_( ctx ), size_( size ), flags_( flags ), hostPtr_( hostPtr )
    {
        AllocateMemObj( size );
    }
    CLMemObj( const CLMemObj& other )
    {
        ctx_ = other.ctx_;
        flags_ = other.flags_;
        hostPtr_ = other.hostPtr_;
        AcquireMemObj( other.memObj_ );
    }
    CLMemObj operator=( const CLMemObj& other )
    {
        ReleaseMemObj();
        ctx_ = other.ctx_;
        flags_ = other.flags_;
        hostPtr_ = other.hostPtr_;
        AcquireMemObj( other.memObj_ );
        return *this;
    }
    ~CLMemObj() { ReleaseMemObj(); }
    cl_mem GetCLMemHandle() const { return memObj_; }
    operator cl_mem() const { return GetCLMemHandle(); }
    void* GetHostPtr() const { return hostPtr_; }
    size_t GetSize() const { return size_; }
    cl_context GetCLContext() const { return ctx_; }
    cl_mem Resize( size_t newSize )
    {
        cl_mem oldMemObj = memObj_;
        ReleaseMemObj();
        AllocateMemObj( newSize );
        return oldMemObj;
    }
private:
    void AllocateMemObj( size_t size )
    {
        cl_int statusCode = CL_SUCCESS + 1;
        memObj_ = ::clCreateBuffer( ctx_, flags_, size, hostPtr_, &statusCode );
        if( statusCode != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clCreateBuffer()" );
        }
        size_ = size;
    }
    void ReleaseMemObj()
    {
        if( ::clReleaseMemObject( memObj_ ) != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clReleaseMemObject()" );
        }
    }
    void AcquireMemObj( cl_mem mo )
    {
        if( ::clRetainMemObject( mo ) != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clRetainMemObject()" );
        }
        memObj_ = mo;
    }
private:
    cl_context ctx_;
    cl_mem memObj_;
    size_t size_;
    cl_mem_flags flags_;
    void* hostPtr_; 
};


//------------------------------------------------------------------------------
///Copy from host memory to device memory.
///\param cq command queue to operate on
///\param mo target memory object handler
///\param pHostData source
///\param blocking set blocking or non blocking operation
///\param offset starting point of copy operation in source memory buffer
///\param size number of bytes to copy: in case the value is zero the size
///    specified in the mo parameter is used
///\throw std::runtime_error.
void CLCopyHtoD( cl_command_queue cq, const void* pHostData, CLMemObj& mo,
                 cl_bool blocking = CL_TRUE, size_t offset = 0, size_t size = 0 );


//------------------------------------------------------------------------------
///Copy from device memory to host memory.
///\param cq command queue to operate on
///\param mo source of copy operation
///\param pHostData target of copy operation
///\param blocking set blocking or non blocking operation
///\param offset starting point of copy operation in source memory buffer
///\param size number of bytes to copy: in case the value is zero the size
///    specified in the mo parameter is used
///\throw std::runtime_error.
void CLCopyDtoH( cl_command_queue cq, const CLMemObj& mo, void* pHostData,
                 cl_bool blocking = CL_TRUE, size_t offset = 0, size_t size = 0 );


/// Type used for local and global workgroup size.
typedef std::vector< size_t > SizeArray;

//------------------------------------------------------------------------------
/// Utility class to setup and run kernels; does not do any resource management
/// it is the resposnsibility of the client code to properly manage the
/// lifecycle of kernel and command queue.
class CLKernelHandler
{
public:
    template < typename CLExecContext > 
    CLKernelHandler( const CLExecContext& ec,
                     const SizeArray& gwgs = SizeArray(),
                     const SizeArray& lwgs = SizeArray() ) : 
        commandQueue_( ec.commandQueue ), kernel_( ec.kernel ),
        gwgs_( gwgs ), lwgs_( lwgs )
    {}
    CLKernelHandler( cl_command_queue cq,
                     cl_kernel k,     
                     const SizeArray& gwgs = SizeArray(),
                     const SizeArray& lwgs = SizeArray() ) : 
        commandQueue_( cq ), kernel_( k ),
        gwgs_( gwgs ), lwgs_( lwgs )
    {}
    CLKernelHandler() {} 
    void SetCommandQueue( cl_command_queue cq ) { commandQueue_ = cq; }
    void SetKernel( cl_kernel k ) { kernel_ = k; }
    void SetGlobalWGroupSize( const SizeArray& gwgs ) { gwgs_ = gwgs; }
    const SizeArray& GetGlobalWGroupSize() const { return gwgs_; }
    const SizeArray& GetLocalWGroupSize() const { return lwgs_; }
    void SetLocalWGroupSize( const SizeArray& lwgs )  { lwgs_ = lwgs; }
    template < typename T > void SetParam( int pos, T val )
    {
        T v = val;
        cl_int status = ::clSetKernelArg( kernel_, pos, sizeof( T ), &v );
        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clSetKernelArg()" );
    }
    void AsyncRun()
    {
        cl_int status = 
            ::clEnqueueNDRangeKernel( commandQueue_, kernel_, gwgs_.size(), 0, &gwgs_[ 0 ], &lwgs_[ 0 ], 0, 0, 0 ); 
        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clEnqueueNDRangeKernel()" );
    }
    void SyncRun()
    {
        cl_int status = 
            ::clEnqueueNDRangeKernel( commandQueue_, kernel_, gwgs_.size(), 0, &gwgs_[ 0 ], &lwgs_[ 0 ], 0, 0, 0 ); 
        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clEnqueueNDRangeKernel()" );
        clFinish( commandQueue_ );
    }
private:
    cl_command_queue commandQueue_;
    cl_kernel kernel_;
    SizeArray gwgs_;
    SizeArray lwgs_;
};

//------------------------------------------------------------------------------
/// Invoke kernel asynchronously.  
cl_event InvokeKernelAsync( cl_command_queue cq,
                            cl_kernel k,
                            const SizeArray& gwgs,
                            const SizeArray& lwgs,
                            const VArgList& valist );

//------------------------------------------------------------------------------
/// Invoke kernel asynchronously.  
inline cl_event InvokeKernelAsync( const CLExecutionContext& ec,
                                   const SizeArray& gwgs,
                                   const SizeArray& lwgs,
                                   const VArgList& valist )
{
    return InvokeKernelAsync( ec.commandQueue,
                              ec.kernel,
                              gwgs,
                              lwgs,
                              valist );
}

//------------------------------------------------------------------------------
/// Invoke kernel synchronously.
cl_event InvokeKernelSync( cl_command_queue cq,
                           cl_kernel k,
                           const SizeArray& gwgs,
                           const SizeArray& lwgs,
                           const VArgList& valist );

//------------------------------------------------------------------------------
/// Invoke kernel synchronously.
inline cl_event InvokeKernelSync( const CLExecutionContext& ec,
                                  const SizeArray& gwgs,
                                  const SizeArray& lwgs,
                                  const VArgList& valist )
{
    return InvokeKernelSync( ec.commandQueue,
                             ec.kernel,
                             gwgs,
                             lwgs,
                             valist );
}

//------------------------------------------------------------------------------
/// Release resources stored in execution context.
/// \param[in,out] ec valid execution context
/// \throw std::runtime_error in case an operation fails.
void ReleaseExecutionContext( CLExecutionContext& ec );


//------------------------------------------------------------------------------
///Profiling information.
class ProfilingInfo
{
public:
    ///Constructor; fills member variables through calls to the <em>clGetEventProfilingInfo</em> function.
    ///@throw std::runtime_error in case OpenCL errors occur
    ProfilingInfo( cl_event e );
    ///@return execution time in milliseconds from start to end of command execution
    double ExecutionTime() const { return ( commandEnd - commandStart ) / 1.E6; }
    ///@return delay in milliseconds between submission and start time
    double Latency() const { return ( commandStart - commandSubmitted ) / 1.E6; }
private:
    cl_ulong commandQueued;
    cl_ulong commandSubmitted;
    cl_ulong commandStart;
    cl_ulong commandEnd;
};
///Change command queue properties to enable profiling.
/// @param cq command queue to operate on
/// @return old command queue property flags
/// @throw std::runtime_error in case of OpenCL errors while setting the property flags.
cl_command_queue_properties EnableProfiling( cl_command_queue cq );
///Change command queue properties to disable profiling.
/// @return old command queue property flags
/// @throw std::runtime_error in case of OpenCL errors while setting the property flags.
cl_command_queue_properties DisableProfiling( cl_command_queue cq );

#endif //GPUPP_H_
