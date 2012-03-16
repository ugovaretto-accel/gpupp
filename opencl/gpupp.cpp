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

#include "gpupp.h"
#include <fstream>
#include <sstream>
#include "OpenCLDeviceInfoTable.h"
#include "OpenCLStatusCodesTable.h"

const OpenCLStatusCodesTable& clERRORS = OpenCLStatusCodesTable::Instance();

namespace {
//-----------------------------------------------------------------------------
/// Loads the content of a text file preserving EOL separators.
/// \param[in] fname absolut path of text file
/// \return string with file content
/// \throw std::runtime_error in case the file cannot be opened
std::string LoadText( const std::string& fname )
{
    std::fstream is( fname.c_str() );
    std::string txt;
    if( !is ) throw std::runtime_error( "Cannot open file: " + fname );
    while( is )
    {
        std::string buf;
        std::getline( is, buf );
        txt += '\n';
        txt += buf;
    }
    return txt;
}
}

//------------------------------------------------------------------------------
DeviceInfoMap QueryDevices( cl_platform_id platformID )
{
    DeviceInfoMap dim;
    cl_uint numDevices = cl_uint();
    cl_device_id device = 0;
    cl_int status = ::clGetDeviceIDs( platformID, CL_DEVICE_TYPE_DEFAULT, 0, 0, &numDevices ); // <- NUM DEVICES ?
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetDeviceIDs(): " + clERRORS[ status ] );
    if( numDevices > 0 )
    {
        typedef std::vector< cl_device_id > DeviceIds;
        DeviceIds devices( numDevices );
        status = ::clGetDeviceIDs( platformID, CL_DEVICE_TYPE_DEFAULT, devices.size(), &devices[ 0 ], 0 );
        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetDeviceIDs() " + clERRORS[ status ] );
        
        const OpenCLDeviceInfoTable& DIT = OpenCLDeviceInfoTable::Instance();
        
        // iterate over platforms and print properties to text buffers
        for( DeviceIds::iterator i = devices.begin(); i != devices.end();  ++i )
        {
            typedef OpenCLDeviceInfoTable::IDIterator IT;
            for( IT di = DIT.DeviceIdBegin(); di != DIT.DeviceIdEnd(); ++di )
            {
                std::ostringstream infoStream;
                std::vector< char > buf( 1 << 14, char() );
                status = ::clGetDeviceInfo( *i, di->first, buf.size(), &buf[ 0 ], 0 );
                if( status != CL_SUCCESS ) continue; /*throw std::runtime_error( 
                                                    "ERROR - clGetDeviceInfo() - " +
                                                    di->second + ": " + clERRORS[ status ] );*/
                if( DIT.IsUInt( di->first ) )
                {
                    if( di->first == CL_DEVICE_PLATFORM )
                    {
                        infoStream << std::ios_base::hex << *reinterpret_cast< unsigned* >( &buf[ 0 ] );
                    }
                    else if( di->first == CL_DEVICE_TYPE )
                    {
                        unsigned dt = *reinterpret_cast< unsigned* >( &buf[ 0 ] );
                        if( dt & CL_DEVICE_TYPE_DEFAULT     ) infoStream << "Default ";
                        if( dt & CL_DEVICE_TYPE_CPU         ) infoStream << " CPU ";
                        if( dt & CL_DEVICE_TYPE_GPU         ) infoStream << " GPU ";
                        if( dt & CL_DEVICE_TYPE_ACCELERATOR ) infoStream << " Accelerator ";
                    }
                    else if( di->first == CL_DEVICE_SINGLE_FP_CONFIG )
                    {
                        unsigned dt = *reinterpret_cast< unsigned* >( &buf[ 0 ] );
                        if( dt & CL_FP_DENORM           ) infoStream << "DENORM ";
                        if( dt & CL_FP_INF_NAN          ) infoStream << " INF_NAN ";
                        if( dt & CL_FP_ROUND_TO_NEAREST ) infoStream << " ROUND_TO_NEAREST ";
                        if( dt & CL_FP_ROUND_TO_ZERO    ) infoStream << " ROUND_TO_ZERO ";
                        if( dt & CL_FP_ROUND_TO_INF     ) infoStream << " ROUND_TO_INF ";
                        if( dt & CL_FP_FMA              ) infoStream << " FMA ";
                    }
                    else
                    {
                        infoStream << *reinterpret_cast< unsigned* >( &buf[ 0 ] );
                    }
                }
                else
                {
                    infoStream << &buf[ 0 ];
                }                

                dim[ di->second ] = infoStream.str();
            }
        }
    }
    return dim;
}


//-----------------------------------------------------------------------------
/// Print device information.
void PrintDeviceInfo( std::ostream& os, const DeviceInfoMap& dim, const std::string& indent = "\t" )
{
    for( DeviceInfoMap::const_iterator i = dim.begin(); i != dim.end(); ++i )
    {
        os << indent << i->first << ": " << i->second << '\n';
    }
}

//------------------------------------------------------------------------------
void PrintPlatformInfo( std::ostream& os, const PlatformInfo& pi,
                        const std::string& deviceIndent)
{
    os << "Platform:  " << pi.name       << '\n';
    os << "Vendor:    " << pi.vendor     << '\n';
    os << "Profile:   " << pi.profile    << '\n';
    os << "Version:   " << pi.version    << '\n';
    os << "Extensions " << pi.extensions << '\n';
    os << "Devices:   " << '\n';
    int di = 0;
    for( Devices::const_iterator i = pi.devices.begin();
         i != pi.devices.end(); ++i )
    {  
        os << "------------------------------\n";
        os << "Device " << di++ << ":\n";
        PrintDeviceInfo( os, *i, deviceIndent );
    }
}


//------------------------------------------------------------------------------
/// Print platform info.
void PrintPlatformsInfo( std::ostream& os, const Platforms& p,
                         const std::string& deviceIndent )
{
    for( Platforms::const_iterator i = p.begin(); i != p.end(); ++i )
    {
        os << "\n=========================================\n";
        PrintPlatformInfo( os, *i, deviceIndent );
    }
}

//------------------------------------------------------------------------------
Platforms QueryPlatforms()
{
    Platforms retPlatforms;
    cl_uint numPlatforms = cl_uint();
    cl_platform_id platform = 0;
    cl_int status = ::clGetPlatformIDs( 0, 0, &numPlatforms ); // <- NUM PLATFORMS ?
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformIDs(): " + clERRORS[ status ]  );
    if( numPlatforms > 0 )
    {
        typedef std::vector< cl_platform_id > PlatformIds;
        PlatformIds platforms( numPlatforms );
        status = ::clGetPlatformIDs( platforms.size(), &platforms[ 0 ], 0 );
        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformIDs(): " + clERRORS[ status ] );
        // iterate over platforms and fill platform info structure
        std::vector< char > buf( 1 << 14, char() );
        for( PlatformIds::iterator i = platforms.begin(); i != platforms.end(); ++i )
        {
            PlatformInfo pi;
            status = ::clGetPlatformInfo( *i, CL_PLATFORM_VENDOR, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            pi.vendor = &buf[ 0 ]; 

            status = ::clGetPlatformInfo( *i, CL_PLATFORM_PROFILE, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            pi.profile = &buf[ 0 ]; 

            status = ::clGetPlatformInfo( *i, CL_PLATFORM_VERSION, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            pi.version = &buf[ 0 ];

            status = ::clGetPlatformInfo( *i, CL_PLATFORM_NAME, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            pi.name = &buf[ 0 ];
            
            status = ::clGetPlatformInfo( *i, CL_PLATFORM_EXTENSIONS, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            pi.extensions = &buf[ 0 ];
            
            pi.devices.push_back( QueryDevices( *i ) );

            retPlatforms.push_back( pi );
        }
    }
    return retPlatforms;
}

//------------------------------------------------------------------------------
CLExecutionContext CreateCLExecutionContext( const std::string& platformString,
                                             int deviceNum,
                                             cl_device_type deviceType  )
{
    if( platformString.size() < 1 )
    {
        throw std::logic_error( "Empty platform string" );
    }
       
    if( deviceNum < 0 )
    {
        throw std::logic_error( "Invalid device number" );
    }

    //SELECT PLATFORM
    cl_uint numPlatforms = cl_uint();
    cl_platform_id platform = cl_platform_id();
    cl_int status = ::clGetPlatformIDs( 0, 0, &numPlatforms );

    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformIDs(): " + clERRORS[ status ] );

    if( numPlatforms > 0 )
    {
        std::vector< cl_platform_id > platforms( numPlatforms );
              
        status = ::clGetPlatformIDs( numPlatforms, &platforms[ 0 ], 0 );

        if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformIDs(): " + clERRORS[ status ] );

        // iterate over platforms and look for `platformString` platform name;
        // right way of doing it is to get a device type info and check that
        // (CL_DEVICE_TYPE & CL_DEVICE_TYPE_GPU != 0) && (CL_DEVICE_AVAILABLE > 0)
        // is true
        for( unsigned i = 0; i !=  numPlatforms; ++i )
        {
            std::vector< char > buf( 256, char() );
            status = ::clGetPlatformInfo( platforms[ i ], CL_PLATFORM_NAME, buf.size(), &buf[ 0 ], 0 );
            if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetPlatformInfo(): " + clERRORS[ status ] );
            if( platformString == &buf[ 0 ] )
            {
                platform = platforms[ i ];
                break;
            }
        }
    }
            
    if( platform == 0 ) throw std::runtime_error( "Couldn't find suitable platform" );
    
    //CREATE CONTEXT
    cl_context_properties ctxProps[] = { CL_CONTEXT_PLATFORM,
                                         reinterpret_cast< cl_context_properties >( platform ),
                                         0 };
    status = CL_SUCCESS - 1;
    
    //create context from specified type (GPU/CPU/DEFAULT/ACCELERATOR...)
    HContext ctx( ::clCreateContextFromType( ctxProps, deviceType, 0, 0, &status ) );

    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clCreateContextFromType(): " + clERRORS[ status ] );

    
    //retrieve device to use:
    //1) retrieve the number of bytes required to store the list of devices
    size_t cd = cl_uint(); // 
    status = ::clGetContextInfo( ctx, CL_CONTEXT_DEVICES, 0, 0, &cd );
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetContextInfo(): " + clERRORS[ status ] );
    //2) create a vector of device ids to store the returned list of devices
    std::vector< cl_device_id > devices( cd / sizeof( cl_device_id ) );
    status = ::clGetContextInfo( ctx, CL_CONTEXT_DEVICES, cd, &devices[ 0 ], 0 );
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetContextInfo(): " + clERRORS[ status ] );
    
    // get device at position specified in parameter list
    if( size_t( deviceNum ) >= cd ) throw std::range_error( "Invalid device index" );
    cl_device_id device = devices[ deviceNum ];
    
    // construct and return context
    return CLExecutionContext( platform, device, ctx );
}

//-----------------------------------------------------------------------------
CLExecutionContext CreateCommandQueue( CLExecutionContext ec, cl_command_queue_properties prop = cl_command_queue_properties() )
{
    if( ec.context == 0 ) throw std::logic_error( "Uninitialized execution context" );
    cl_int status = CL_SUCCESS + 1;
    cl_command_queue cq = ::clCreateCommandQueue( ec.context, ec.device, prop, &status ); // no properties defined, use default
    ec.commandQueue = HCommandQueue( cq );
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clCreateCommandQueue(): " + clERRORS[ status ] );
    return ec;
}

//-----------------------------------------------------------------------------
CLExecutionContext BuildKernel( CLExecutionContext ec,
                                const std::string& kernelSrc,
                                const std::string& kernelName,
                                std::string& buildOutput,
                                const std::string& buildOptions,
                                bool computeWGroupSize )
{

    assert( kernelSrc.size() > 0 );
    assert( kernelName.size() > 0 );
    if( ec.context == 0 ) throw std::logic_error( "Uninitialized execution context" );
    
    cl_int status = CL_SUCCESS + 1;
    //CREATE PROGRAM
    const size_t kernelSrcLength = kernelSrc.size();
    const char* src = kernelSrc.c_str();
    ec.program = HProgram( clCreateProgramWithSource( ec.context, 1, &src, &kernelSrcLength, &status ) );
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clCreateProgramWithSource(): " + clERRORS[ status ] );
    
    //BUILD PROGRAM
    cl_int buildStatus = clBuildProgram( ec.program, 1, &ec.device, buildOptions.c_str(), 0, 0 );
    //log output if any
    char buffer[1 << 14] = "";
    size_t len = 0;
    status = ::clGetProgramBuildInfo( ec.program, ec.device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetProgramBuildInfo(): " + clERRORS[ status ] );
    if( buffer[ 0 ] != 0 ) buildOutput = &buffer[ 0 ]; 
    if( buildStatus != CL_SUCCESS ) throw std::runtime_error( "ERROR - clGetBuildProgram: " + clERRORS[ status ] + "\n" + buildOutput );
    
    //CREATE KERNEL
    ec.kernel = HKernel( clCreateKernel( ec.program, kernelName.c_str(), &status ) );
    if( status != CL_SUCCESS ) throw std::runtime_error( "ERROR - clCreateKernel(): " + clERRORS[ status ] );

    //RETRIEVE KERNEL INFO
    size_t returnedValueSize = 0;
    //on some cards the following value is not returned
    status = clGetKernelWorkGroupInfo( ec.kernel, ec.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof( size_t ), (void* ) &ec.wgroupSize, &returnedValueSize );
    if( status != CL_SUCCESS && status != CL_INVALID_VALUE ) throw std::runtime_error( "ERROR - clGetKernelWorkGroupInfo(): " + clERRORS[ status ] );
    status = clGetKernelWorkGroupInfo( ec.kernel, ec.device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof( size_t ), (void* ) &ec.localMemSize, &returnedValueSize );
    if( status != CL_SUCCESS && status != CL_INVALID_VALUE ) throw std::runtime_error( "ERROR - clGetKernelWorkGroupInfo(): " + clERRORS[ status ] );
    
    return ec;
}


//-----------------------------------------------------------------------------
CLExecutionContext CreateContextAndKernel( const std::string& platformString,
                                           cl_device_type deviceType,
                                           int deviceNum,     
                                           const std::string& kernelSrc,
                                           const std::string& kernelName,
                                           std::string& buildOutput,
                                           const std::string& buildOptions,
                                           bool computeWGroupSize,										   
										   cl_command_queue_properties prop )
{
    return
        BuildKernel(
            CreateCommandQueue(
                CreateCLExecutionContext( platformString, deviceNum, deviceType ), prop ),
            kernelSrc,
            kernelName,
            buildOutput, 
            buildOptions,
            computeWGroupSize );
}

//-----------------------------------------------------------------------------
CLExecutionContext CreateContextAndKernelFromFile( const std::string& platformString,
                                                   cl_device_type deviceType,
                                                   int deviceNum,
                                                   const std::string& kernelPath,
                                                   const std::string& kernelName,
                                                   std::string& buildOutput,
                                                   const std::string& buildOptions,
                                                   bool computeWGroupSize,
												   cl_command_queue_properties prop )
{
    return CreateContextAndKernel( platformString,
                                   deviceType,
                                   deviceNum,
                                   LoadText( kernelPath ),
                                   kernelName,
                                   buildOutput,
                                   buildOptions,
                                   computeWGroupSize,
								   prop );
}

//------------------------------------------------------------------------------
void CLCopyHtoD( cl_command_queue cq, CLMemObj& mo, const void* pHostData, cl_bool blocking, size_t offset, size_t size )
{
    cl_int status = CL_SUCCESS + 1;
    if( size == 0 && offset == 0 )
    {
        status = ::clEnqueueWriteBuffer( cq, mo.GetCLMemHandle(), blocking, 0, mo.GetSize(), pHostData, 0, 0, 0 );
        if( status != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clEnqueueWriteBuffer(): " + clERRORS[ status ] );
        }
    }
    else
    {
        if( mo.GetSize() - offset < size )
        {
            throw std::logic_error( "Error - destination buffer smaller than data size" ); 
        }
        if( ::clEnqueueWriteBuffer( cq, mo.GetCLMemHandle(), blocking, offset, size, pHostData, 0, 0, 0 )
            != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clEnqueueWriteBuffer(): " + clERRORS[ status ] );
        }
    }
    if( blocking == CL_TRUE )
    {
        status = ::clFinish( cq );
        if(  status != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clFinish(): " + clERRORS[ status ] );
        }
    }
}

//------------------------------------------------------------------------------
void CLCopyDtoH( cl_command_queue cq, const CLMemObj& mo, void* pHostData, cl_bool blocking, size_t offset, size_t size )
{
    cl_int status = CL_SUCCESS + 1;
    if( size == 0 && offset == 0 )
    {
        status = ::clEnqueueReadBuffer( cq, mo.GetCLMemHandle(), blocking, 0, mo.GetSize(), pHostData, 0, 0, 0 );
        if( status != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clEnqueueReadBuffer(): " + clERRORS[ status ] );
        }
    }
    else
    {
        if( mo.GetSize() - offset > size )
        {
            throw std::logic_error( "Error - destination buffer smaller than data size" ); 
        }
        if( ::clEnqueueReadBuffer( cq, mo.GetCLMemHandle(), blocking, offset, size, pHostData, 0, 0, 0 )
            != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clEnqueueReadBuffer(): " + clERRORS[ status ] );
        }
    }
    if( blocking == CL_TRUE )
    {
        status = ::clFinish( cq );
        if(  status != CL_SUCCESS )
        {
            throw std::runtime_error( "Error - clFinish(): " + clERRORS[ status ] );
        }
    }
}


//------------------------------------------------------------------------------
cl_event InvokeKernelAsync( cl_command_queue cq,
                        cl_kernel k,
                        const SizeArray& gwgs,
                        const SizeArray& lwgs,
                        const VArgList& valist )
{
    size_t pos = 0;
    for( VArgList::ArgListConstIterator i = valist.Begin(); i != valist.End(); ++i, ++pos )
    {
        cl_int status = CL_SUCCESS;
        if( status = ::clSetKernelArg( k, pos, AnySizeOf( *i ), AnyAddress( *i ) ) != CL_SUCCESS )
        {
            
            throw std::runtime_error( "ERROR - clSetKernelArg(): " + clERRORS[ status ] );
        }
    }
    cl_event clevent = cl_event();
    cl_int status = CL_SUCCESS + 1;
    status = ::clEnqueueNDRangeKernel( cq, 
                                       k,
                                       gwgs.size(),
                                       0,
                                       &gwgs[ 0 ],
                                       &lwgs[ 0 ], 0, 0, &clevent );
    if(  status != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clEnqueueNDRangeKernel(): " + clERRORS[ status ] );
    }
    if( ::clFlush( cq ) != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clFlush(): " + clERRORS[ status ] );
    }

    return clevent;
}    

//------------------------------------------------------------------------------
cl_event InvokeKernelSync(  cl_command_queue cq,
                        cl_kernel k,
                        const SizeArray& gwgs,
                        const SizeArray& lwgs,
                        const VArgList& valist )
{
    cl_event clevent = InvokeKernelAsync( cq, k, gwgs, lwgs, valist );
    cl_int status = CL_SUCCESS + 1;
    if( status = ::clFinish( cq ) != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clFinish(): " + clERRORS[ status ] );
    }

    return clevent;
}

//------------------------------------------------------------------------------
void ReleaseExecutionContext( CLExecutionContext& ec )
{
    ec.commandQueue.Release();
    ec.program.Release();
    ec.kernel.Release();
    ec.context.Release();
}

//------------------------------------------------------------------------------
ProfilingInfo::ProfilingInfo( cl_event e )
{
    cl_int status = ::clGetEventProfilingInfo( e,
                                               CL_PROFILING_COMMAND_QUEUED,
                                               sizeof( commandQueued ),
                                               &commandQueued,
                                               0 );
    if(  status != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clGetEventProfilingInfo(): " + clERRORS[ status ] );
    }
    status = ::clGetEventProfilingInfo( e,
                                            CL_PROFILING_COMMAND_SUBMIT,
                                            sizeof( commandSubmitted ),
                                            &commandSubmitted,
                                            0 );
    if(  status != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clGetEventProfilingInfo(): " + clERRORS[ status ] );
    }
    status = ::clGetEventProfilingInfo( e,
                                        CL_PROFILING_COMMAND_START,
                                        sizeof( commandStart ),
                                        &commandStart,
                                        0 );
    if(  status != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clGetEventProfilingInfo(): " + clERRORS[ status ] );
    }
    status = ::clGetEventProfilingInfo( e,
                                        CL_PROFILING_COMMAND_END,
                                        sizeof( commandEnd ),
                                        &commandEnd,
                                        0 );
    if(  status != CL_SUCCESS )
    {
        throw std::runtime_error( "ERROR - clGetEventProfilingInfo(): " + clERRORS[ status ] );
    }
}


// OpenCL 1.1 has deprecated clSetCommandQueueProperty() because
//it isn't thread safe.  Rather than have a difference between the
//OpenCL 1.0 and 1.1 versions of QtOpenCL, we just implement
//the OpenCL 1.1 behavior.
//Out-of-order execution and profiling can be enabled by explicitly
// calling CreateCommandQueue(...,properties).
//cl_command_queue_properties EnableProfiling( cl_command_queue cq )
//{
//    cl_command_queue_properties cp = cl_command_queue_properties();
//    cl_int status = ::clSetCommandQueueProperty( cq, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, &cp );
//    if(  status != CL_SUCCESS )
//    {
//        throw std::runtime_error( "ERROR - clSetCommandQueueProperty(): " + clERRORS[ status ] );
//    }
//    return cp;
//}
//cl_command_queue_properties DisableProfiling( cl_command_queue cq )
//{
//    cl_command_queue_properties cp = cl_command_queue_properties();
//    cl_int status = ::clSetCommandQueueProperty( cq, CL_QUEUE_PROFILING_ENABLE, CL_FALSE, &cp );
//    if(  status != CL_SUCCESS )
//    {
//        throw std::runtime_error( "ERROR - clSetCommandQueueProperty(): " + clERRORS[ status ] );
//    }
//    return cp;
//}
