///\file cuda/gpupp.h CUDA functions and types

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

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <iterator>
#include <cuda.h>
#include "../utility/varargs.h"
#include "../utility/ResourceHandler.h"

/// CUDA timer: Uses events to time execution
class CUDATimer
{
public:
	/// Constructor.
	/// \param hStream stream to attach event to, deafault is zero
	/// \param flags event flags; default is CU_EVENT_DEFAULT
	/// \throw std::runtime_error in case of problems creating events
	CUDATimer( CUstream hStream = 0, unsigned int flags = CU_EVENT_DEFAULT ) : hStream_( hStream ), elapsedTime_( 0.f )
	{
		if( ::cuEventCreate( &start_, flags ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cuEventCreate() - start" );
		}
		if( ::cuEventCreate( &stop_, flags ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cuEventCreate() - stop" );
		}
	}
	/// Start timer by recording event
	/// \throw std::runtime_error in case record of start event fails
	void Start() 
	{ 
		if( ::cuEventRecord( start_, hStream_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cudaEventRecord() - start" );
		}
	}
	/// Stop timer by recording event execution and synchronizing to wait for
	/// end of operations associated with stream
	/// \return elapsed time
	/// \throw std::runtime_error in case record of stop event or synchronization fail
	float Stop() 
	{ 
		if( ::cuEventRecord( stop_, hStream_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cudaEventRecord() - stop" );
		}
		if( ::cuEventSynchronize( stop_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cudaEventSynchronize()" );
		}
		QueryElapsedTime();
		return ElapsedTime();
	}
	/// Returns elapsed time; Start() and Stop() must have been invoked prior
	/// to invocation of this method
	/// \return elapsed time
	float ElapsedTime() const { return elapsedTime_; }
	/// Destructor throwing exceptions! This is an RAII class anyway
	/// \throw std::runtime_error in case event destruction fails 
	~CUDATimer() // throws exceptions!
	{
		if( ::cuEventDestroy( start_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cuEventDestroy() - start" );
		}
        if( ::cuEventDestroy( stop_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cuEventDestroy() - stop" );
		}
	}
private:
	/// Query CUDA run-time for elapsed time
	void QueryElapsedTime()
	{
		if( ::cuEventQuery( start_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Start event not recorded" );
		}
		if( ::cuEventQuery( stop_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Stop event not recorded" );
		}
		if( ::cuEventElapsedTime( &elapsedTime_, start_, stop_ ) != CUDA_SUCCESS )
		{
			throw std::runtime_error( "Error - cuEventElapsedTime()" );
		}
	}
private:
	/// Stream associated with excution
	CUstream hStream_;
	/// Elapsed time returned by CUDA run-time
	float elapsedTime_;
	/// Start event
	CUevent start_;
	/// Stop event
	CUevent stop_;
};


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
///Program(Module) name
struct ProgramName
{
    operator const char*() const { return "Program"; }
};

///Note that on WIN32 it is required to set the calling convention
///of resource handling functions to \c __stdcall
#if defined( _WIN32 ) && !defined( _WIN64 )
#define GPUPP_CALL_MODIFIER __stdcall
#else
#define GPUPP_CALL_MODIFIER
#endif

///Null function implementation.
template < class T, class R > 
 R GPUPP_CALL_MODIFIER NullFun( T ) { return R(); }


typedef CUfunction CUkernel;

///Handler for context resources
typedef ResourceHandler< CUcontext,
                         CUresult,
                         NullFun< CUcontext, CUresult >,
                         ::cuCtxDestroy,
                         ContextName,
                         CUDA_SUCCESS > HContext;
///Handler for kernel resources
typedef ResourceHandler< CUkernel,
                         CUresult,
                         NullFun< CUkernel, CUresult >,
                         NullFun< CUkernel, CUresult >,
                         KernelName,
                         CUDA_SUCCESS > HKernel;
///Handler for module (program) resources.
typedef ResourceHandler< CUmodule,
                         CUresult,
                         NullFun< CUmodule, CUresult >,
                         ::cuModuleUnload,
                         ProgramName,
                         CUDA_SUCCESS > HProgram; 

//------------------------------------------------------------------------------
///CUDA device information.
struct CUDADeviceInfo
{
    typedef std::map< std::string, int > AttributeMap;
    ///Device's computing capabilities major version number.
    int cdMajor;
    ///Device's computing capabilities minor version number.
    int cdMinor;
    ///Device name.
    std::string name;
    ///Driver major version number.
    int driverVersionMajor;
    ///Driver minor version number.
    int driverVersionMinor;
    ///Size of global memory available on device.
    size_t totalMem;
    ///Key,value pair of attributes; keys are string
    ///representing the attribute identifier's name.
    AttributeMap attributeMap;
    CUDADeviceInfo() : cdMajor( 0 ), cdMinor( 0 ),
        driverVersionMajor( 0 ), driverVersionMinor( 0 ), totalMem( 0 )
    {}
};
///Overloaded operator to print CUDADeviceInfo content.
inline std::ostream& operator <<( std::ostream& os, const CUDADeviceInfo& di )
{
    os << "Name:                   " << di.name << '\n';
    os << "Driver version:         " << di.driverVersionMajor << '.' << di.driverVersionMinor << '\n';
    os << "Computing capabilities: " << di.cdMajor << '.' << di.cdMinor << '\n';
    os << "Total memory:           " << di.totalMem << '\n';
    os << "Attributes:" << '\n';
    for( CUDADeviceInfo::AttributeMap::const_iterator i = di.attributeMap.begin();
        i != di.attributeMap.end(); ++i )
    {
        os << "  " << i->first << " = " << i->second << '\n';
    }
    return os;
}

typedef std::vector< CUDADeviceInfo > DeviceInfoArray;

///Returns a sequence of device info records.
DeviceInfoArray QueryDevicesInfo();

///Utility function to print a sequence of CUDADeviceInfo instances to an output stream.
inline void PrintDevicesInfo( std::ostream& os, const DeviceInfoArray& dia )
{
    std::copy( dia.begin(), dia.end(), std::ostream_iterator< CUDADeviceInfo >( os, "================\n\n" ) );
}

//-----------------------------------------------------------------------------
/// Execution context with complete information on execution environment.
/// This class groups together in a bundle context, command queue, and kernel.
/// Automatic resource management is performed by the data members.
struct CUDAExecutionContext
{
    /// CUDA computing device id
    CUdevice device;
    /// CUDA context
    HContext context;
    /// CUDA module
    HProgram program;
    /// CUDA function (kernel)
    HKernel kernel;
    /// Default constructor
    CUDAExecutionContext() : device( 0 ) {}
    /// Constructor as used in CreateCLContext function.
    CUDAExecutionContext( CUdevice d,
                          CUcontext ctx ) : 
                          device( d ), context( ctx ) {}
    
};

typedef HProgram HModule;

//------------------------------------------------------------------------------
/// Create CUDA Context.
/// \param[in] deviceNum index of device to use
/// \param[in] flags flags passed to CUDA context creation function
/// \return valid execution context
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CUDAExecutionContext CreateCUExecutionContext( int deviceNum, unsigned flags = 0 );

//-----------------------------------------------------------------------------
/// Build kernel from source text. Valid program, kernel and info are added
/// into a copy of the passed context.
/// \param[in] ec valid execution context
/// \param[in] kernelSrc  source code of program
/// \param[in] kernelName name of kernel function
/// \return copy of passed execution context with added program and kernel handles
/// \throw std::runtime_error in case of errors while invoking OpenCL functions
CUDAExecutionContext BuildKernel( CUDAExecutionContext ec,
                                  const std::string& kernelSrc,
                                  const std::string& kernelName );

//-----------------------------------------------------------------------------
/// Wrapper function that simply calls CreateCUExecutionContext and
/// BuildKernel in sequence.
/// \param[in] deviceNum index of device to use
/// \param[in] kernelSrc source code of program
/// \param[in] kernelName name of kernel function
/// \param[in] flags flags passed to CUDA context creation function
/// \return valid CUDAExecutionContext with all members initialized
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CUDAExecutionContext CreateContextAndKernel( int deviceNum,     
                                             const std::string& kernelSrc,
                                             const std::string& kernelName,
                                             unsigned flags = 0);

//-----------------------------------------------------------------------------
/// Wrapper function that simply calls CreateContextAndKernel, which in turns
/// calls CreateCUExecutionContext and BuildKernel.
/// \param[in] deviceNum index of device to use
/// \param[in] kernelPath full path to file containing source code of OpenCL program
/// \param[in] kernelName name of kernel function
/// \param[in] flags flags passed tu CUDA contxt creation function
/// \return valid CUDAExecutionContext with all members initialized
/// \throw std::runtime_error in case of failure to allocate resources
/// \throw std::range_error in case the index of the device is out if bounds
CUDAExecutionContext CreateContextAndKernelFromFile( int deviceNum,
                                                     const std::string& kernelPath,
                                                     const std::string& kernelName,
                                                     unsigned flags = 0);

//------------------------------------------------------------------------------
/// Wrapper for CUDA memory object which performs automatic resource
/// deallocation and reference counting. Note that CUDA does not
/// implementing reference counting for mem objects!!
class CUMemObj
{
    typedef SimpleCounter Counter; //need to perform reference counting
                                   //since cuda does not have reference counted mem objects
    CUMemObj(); // do not allow default construction; force client
                // code to specify valid context
public:
    CUMemObj( CUcontext ctx, // force to specify context
              unsigned size, 
              void* hostPtr = 0 ) 
              : ctx_( ctx ), size_( size ), hostPtr_( hostPtr ), counter_( new Counter )
    {
        AllocateMemObj( size );
    }
    CUMemObj( const CUMemObj& other )
    {
        counter_ = other.counter_;
        ctx_ = other.ctx_;
        hostPtr_ = other.hostPtr_;
        size_ = other.size_;
        AcquireMemObj( other.memObj_ );
    }
    CUMemObj operator=( const CUMemObj& other )
    {
        ReleaseMemObj();
        counter_ = other.counter_;
        ctx_ = other.ctx_;
        size_ = other.size_;
        hostPtr_ = other.hostPtr_;
        AcquireMemObj( other.memObj_ );
        return *this;
    }
    ~CUMemObj() { ReleaseMemObj(); }
    CUdeviceptr GetCUMemHandle() const { return memObj_; }
    operator CUdeviceptr() const { return GetCUMemHandle(); }
    void* GetHostPtr() const { return hostPtr_; }
    size_t GetSize() const { return size_; }
    CUcontext GetCUContext() const { return ctx_; }
    CUdeviceptr Resize( unsigned newSize )
    {
        CUdeviceptr oldMemObj = memObj_;
        ReleaseMemObj();
        AllocateMemObj( newSize );
        return oldMemObj;
    }
    friend std::ostream& operator <<( std::ostream& os, const CUMemObj& mo )
    {
        os << mo.GetCUMemHandle();
        return os;
    }
private:
    void AllocateMemObj( unsigned bytesize )
    {
        // page locked ?
        if( hostPtr_ != 0 )
        {
            throw std::invalid_argument( "Page locked allocation not implemented yet" );
        }
        else
        {
            CUresult status = ::cuMemAlloc( &memObj_, bytesize );
            if( status != CUDA_SUCCESS )
            {
                throw std::runtime_error( "Error - cuMemAlloc()" );
            }
            size_ = bytesize;
        }
        
    }
    void ReleaseMemObj()
    {
        if( !counter_ ) return;
        if( counter_->Dec() == 0 )
        {
            CUresult status = ::cuMemFree( memObj_ );
            if( status != CUDA_SUCCESS )
            {
                throw std::runtime_error( "Error - cuMemFree()" );
            }
            delete counter_;
            counter_ = 0;
        }        
    }
    void AcquireMemObj( CUdeviceptr mo )
    {
        memObj_ = mo;
        counter_->Inc();        
    }
private:
    CUcontext ctx_;
    CUdeviceptr memObj_;
    size_t size_;
    void* hostPtr_;
    Counter* counter_;
};



//------------------------------------------------------------------------------
///Copy from host memory to device memory.
///\param mo target memory object handler
///\param pHostData source
///\param blocking set blocking or non blocking operation
///\param size number of bytes to copy: in case the value is zero the size
///    specified in the mo parameter is used
///\throw std::runtime_error.
void CUDACopyHtoD( CUMemObj& mo, const void* pHostData, bool blocking = true,
                   unsigned size = 0 );


//------------------------------------------------------------------------------
///Copy from device memory to host memory.
///\param mo source of copy operation
///\param pHostData target of copy operation
///\param blocking set blocking or non blocking operation
///\param size number of bytes to copy: in case the value is zero the size
///    specified in the mo parameter is used
///\throw std::runtime_error.
void CUDACopyDtoH( const CUMemObj& mo, void* pHostData, bool blocking = true,
                   unsigned size = 0 );


//NO EASY WAY OF IMPLEMENTING A KERNEL HANDLER IN CUDA SINCE PARAMETERS CANNOT BE
//SET BY POSITION BUT REQUIRE COMPUTATION OF THE PROPER OFFSET 
//ONE OPTION COULD BE TO FORCE CLIENT CODE TO PASS A SAMPLE ARGUMENT
//LIST TO THE CONSTRUCTOR TO BE USED TO INVOKE KERNELS

/// Type used for local and global workgroup size.
typedef std::vector< size_t > SizeArray;

//------------------------------------------------------------------------------
struct CUMemLayout
{
    SizeArray gridShape;
    SizeArray threadBlockShape;
    CUMemLayout( const SizeArray& gs, const SizeArray& tbs )
        : gridShape( gs ), threadBlockShape( tbs )
    {
        if( gs.size() != 2 ) throw std::range_error( "Error - grid shape must be 2D");
        if( tbs.size() != 3 ) throw std::range_error( "Error - thread block shape must be 3D" );
    }
private:
    CUMemLayout();
};


//------------------------------------------------------------------------------
/// Returns a CUDA style (2D grid of 3D groups!) memory layout given a 
/// domain and thread block shape.
CUMemLayout ComputeCUDAMemLayout( const SizeArray& gwgs,
                                  const SizeArray& lwgs ); 


//------------------------------------------------------------------------------
/// Invoke kernel asynchronously.  
CUMemLayout InvokeKernelAsync( CUkernel k,
                               const SizeArray& gwgs,
                               const SizeArray& lwgs,
                               const VArgList& valist );

//------------------------------------------------------------------------------
/// Invoke kernel asynchronously.  
inline void InvokeKernelAsync( const CUDAExecutionContext& ec,
                               const SizeArray& gwgs,
                               const SizeArray& lwgs,
                               const VArgList& valist )
{
    InvokeKernelAsync(  ec.kernel,
                        gwgs,
                        lwgs,
                        valist );
}

//------------------------------------------------------------------------------
/// Invoke kernel synchronously.
CUMemLayout InvokeKernelSync( CUkernel k,
                                const SizeArray& gwgs,
                              const SizeArray& lwgs,
                              const VArgList& valist );

//------------------------------------------------------------------------------
/// Invoke kernel synchronously.
inline void InvokeKernelSync( const CUDAExecutionContext& ec,
                              const SizeArray& gwgs,
                              const SizeArray& lwgs,
                              const VArgList& valist )
{
    InvokeKernelSync( ec.kernel,
                      gwgs,
                      lwgs,
                      valist );
}

//------------------------------------------------------------------------------
/// Release resources stored in execution context.
/// \param[in,out] ec valid execution context
/// \throw std::runtime_error in case an operation fails.
void ReleaseExecutionContext( CUDAExecutionContext& ec );

#endif //GPUPP_H_
