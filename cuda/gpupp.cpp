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
#include "../utility/alignment.h"
#include <fstream>
#include <sstream>
#include "CUDADeviceInfoTable.h"
#include "CUDAStatusCodesTable.h"

const CUDAStatusCodesTable& cuERRORS = CUDAStatusCodesTable::Instance();

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
// According to NVIDIA sample code 999 means unrecognizable/undetactable/unavailable device (!?)
const int NO_CUDA_DEVICE_MAJOR = 999;
const int NO_CUDA_DEVICE_MINOR = 999;
DeviceInfoArray QueryDevicesInfo()
{
#if CUDA_VERSION < 2020
  #error "CUDA VERSION >= 2.0.2 required"
#endif
	CUresult status = ::cuInit(0); //this plain sucks how are libraries going to deal with this ?
		                           //is it possible to call it multiple times without side effects ?

    if( status != CUDA_SUCCESS )
    {
        throw std::runtime_error( "Error - cuInit(): " + cuERRORS[ status ] );
    }

    const CUDADeviceInfoTable& DIT = CUDADeviceInfoTable::Instance();
    DeviceInfoArray dia;
    int numDevices = 0;
    CUdevice dev = CUdevice();
    status = ::cuDeviceGetCount( &numDevices );
	if( status != CUDA_SUCCESS )
	{
        throw std::runtime_error( "Error - cuDeviceGetCount() " + cuERRORS[ status ] );
	}
	//??? taken form NVIDIA sample code: CUdevice compared with int ? what's the point of the typedef ?
    for( dev = 0; dev < numDevices; ++dev )
    {
		int major = 0;
		int minor = 0;
		status = ::cuDeviceComputeCapability( &major, &minor, dev );
		if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuDeviceComputeCapability() " + cuERRORS[ status ] );
        }
		if( major == NO_CUDA_DEVICE_MAJOR && minor == NO_CUDA_DEVICE_MINOR ) continue;
        std::vector< char > buf(1024, 0);
		status = ::cuDeviceGetName( &buf[ 0 ], int( buf.size() ), dev );
		if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuDeviceGetName() " + cuERRORS[ status ] );
        }
        CUDADeviceInfo di;
        di.cdMajor = major;
        di.cdMinor = minor;
        di.name = std::string( &buf[ 0 ] );
        int iparam = 0;
        size_t uparam = 0;
		status = ::cuDriverGetVersion( &iparam );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuDriverGetVersion() " + cuERRORS[ status ] );            
        }
        di.driverVersionMajor = iparam / 1000;
        di.driverVersionMinor = iparam % 100;
        // total mem is an unsigned int, how are we going to address more than 4GB ??
        status = ::cuDeviceTotalMem( &uparam, dev );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuDeviceTotalMem() " + cuERRORS[ status ] );            
        }
        di.totalMem = uparam;
        for( CUDADeviceInfoTable::IDIterator i = DIT.DeviceIdBegin();
             i != DIT.DeviceIdEnd();
             ++i )
        {
            status = ::cuDeviceGetAttribute( &iparam, i->first, dev );
            if( status != CUDA_SUCCESS )
            {
                throw std::runtime_error( "Error - cuDeviceGetAttribute() " + cuERRORS[ status ] );
            }
            di.attributeMap[ i->second ] = iparam;
        }
        dia.push_back( di );
    }
    return dia;
}


//------------------------------------------------------------------------------
CUDAExecutionContext CreateCUExecutionContext( int deviceNum, unsigned flags )
{
       
    if( deviceNum < 0 )
    {
        throw std::logic_error( "Error - Invalid device number" );
    }

    CUresult status = ::cuInit( 0 ); // is it ok to call this multiple times ?
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuInit() " + cuERRORS[ status ] );
    int deviceCount = 0;
    status = cuDeviceGetCount( &deviceCount );
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuDeviceCount() " + cuERRORS[ status ] );
    if( deviceNum >= deviceCount ) throw std::range_error( "Error - invalid device number" );
    CUdevice device = CUdevice();
    status = cuDeviceGet( &device, 0 );
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuDeviceGet() " + cuERRORS[ status ] );
    CUcontext ctx = CUcontext();
    status = cuCtxCreate( &ctx, flags, device );
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuCtxCreate() " + cuERRORS[ status ] ); 
    return CUDAExecutionContext( device, ctx );
}

//-----------------------------------------------------------------------------
CUDAExecutionContext BuildKernel( CUDAExecutionContext ec,
                                  const std::string& kernelSrc,
                                  const std::string& kernelName )
{

    assert( kernelSrc.size() > 0 );
    assert( kernelName.size() > 0 );
    if( ec.context == 0 ) throw std::logic_error( "Uninitialized execution context" );
        
    //CREATE PROGRAM (MODULE)
    if( kernelSrc.size() == 0 ) throw std::runtime_error( "Error - cannot load kernel" );
    const char* src = kernelSrc.c_str();
    CUmodule module = CUmodule();
    CUresult status = ::cuModuleLoadData( &module, reinterpret_cast< const void* >( src ) );
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuModuleLoadData() " + cuERRORS[ status ]  );
    ec.program = HModule( module ); // wrapped with resource handling code    
    //RETRIEVE KERNEL
    CUfunction kernel = CUfunction();
    status = ::cuModuleGetFunction( &kernel, module, kernelName.c_str() );
    if( status != CUDA_SUCCESS ) throw std::runtime_error( "Error - cuModuleGetFunction(): " + kernelName + cuERRORS[ status ] );
    ec.kernel = HKernel( kernel ); // wrapped with resource handling code   
    return ec;
}

//-----------------------------------------------------------------------------
CUDAExecutionContext CreateContextAndKernel( int deviceNum,     
                                             const std::string& kernelSrc,
                                             const std::string& kernelName,
                                             unsigned flags )
{
    return
        BuildKernel(
            CreateCUExecutionContext( deviceNum, flags ),
            kernelSrc,
            kernelName );
}

//-----------------------------------------------------------------------------
CUDAExecutionContext CreateContextAndKernelFromFile( int deviceNum,
                                                       const std::string& kernelPath,
                                                     const std::string& kernelName,
                                                     unsigned flags )
{
    return CreateContextAndKernel( deviceNum,
                                   LoadText( kernelPath ),
                                   kernelName,
                                   flags );
}

//------------------------------------------------------------------------------
//NOTE: CANNOT SPECIFY DESTINATION OFFSET WITH CUDA!!!! 
void CUDACopyHtoD( CUMemObj& mo, const void* pHostData, bool blocking, unsigned size )
{

    if( blocking )
    {
        CUresult status = ::cuMemcpyHtoD( mo.GetCUMemHandle(), pHostData, size == 0 ? mo.GetSize() : size );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuMemcpyHtoD() " + cuERRORS[ status ] );
        }
    }
    else
    {
        CUresult status = ::cuMemcpyHtoDAsync( mo.GetCUMemHandle(), pHostData, size == 0 ? mo.GetSize() : size, 0 );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuMemcpyHtoDAsync() " + cuERRORS[ status ] );
        }
    }    
}

//------------------------------------------------------------------------------
//NOTE: CANNOT SPECIFY DESTINATION OFFSET WITH CUDA!!!! 
void CUDACopyDtoH( const CUMemObj& mo, void* pHostData, bool blocking, unsigned size )
{

    if( blocking )
    {
        CUresult status = ::cuMemcpyDtoH( pHostData, mo.GetCUMemHandle(), size == 0 ? mo.GetSize() : size );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuMemcpyDtoH() " + cuERRORS[ status ] );
        }
    }
    else
    {
        CUresult status = ::cuMemcpyDtoHAsync( pHostData, mo.GetCUMemHandle(), size == 0 ? mo.GetSize() : size, 0 );
        if( status != CUDA_SUCCESS )
        {
            throw std::runtime_error( "Error - cuMemcpyDtoHAsync() " + cuERRORS[ status ] );
        }
    }    
}

//------------------------------------------------------------------------------
void SetupKernelParameters( CUkernel k, const VArgList& valist )
{
    int offset = 0;
    size_t lastParamSize = 0;
    for( VArgList::ArgListConstIterator i = valist.Begin(); i != valist.End(); ++i )
    {
        if( i->Type() == typeid( CUMemObj ) )
        {
            // SUGGESTED BY NVIDIA (without cast operators): CHECK THEIR EXAMPLES!!!
            CUdeviceptr devptr = CUMemObj( *i ).GetCUMemHandle();
            void* ptr = reinterpret_cast< void* >( size_t( devptr ) );
            offset = AlignedOffset( offset, Alignment( ptr ) );
            CUresult status = ::cuParamSetv( k, offset, &ptr, sizeof( void* ) );
            if( status != CUDA_SUCCESS ) 
            {
                throw std::runtime_error( "Error - cuSetParameterv() " + cuERRORS[ status ] );
            }
            lastParamSize = sizeof( void* );
        }
        else if( i->Type() == typeid( float ) )
        {
            offset = AlignedOffset( offset, ALIGNMENT( float ) );
            CUresult status = ::cuParamSetf( k, offset, *i );
            if( status != CUDA_SUCCESS ) 
            {
                throw std::runtime_error( "Error - cuSetParameterf() " + cuERRORS[ status ] );
            }
            lastParamSize = sizeof( float );
        }
        else if( i->Type() == typeid( unsigned ) )
        {
            offset = AlignedOffset( offset, ALIGNMENT( unsigned ) );
            CUresult status = ::cuParamSeti( k, offset, *i );
            if( status != CUDA_SUCCESS ) 
            {
                throw std::runtime_error( "Error - cuSetParameteri() " + cuERRORS[ status ] );
            }
            lastParamSize = sizeof( unsigned );
        }
        else
        {
            throw std::logic_error( "Error - unrecognized CUDA type: " + std::string( i->Type().name() ) );
        }
        offset += int( lastParamSize );
    }
    ::cuParamSetSize( k, offset );    
}


//------------------------------------------------------------------------------
CUMemLayout ComputeCUDAMemLayout( const SizeArray& gwgs,
                                  const SizeArray& lwgs )
{
    //!! CUDA does not allow 3D grids but does
    // allow 3D thread blocks ?!
    if( gwgs.size() > 3 )
    {
        throw std::logic_error( "Error - only 1,2,3D global domain size allowed" ); 
    }
    if( lwgs.size() > 3 )
    {
        throw std::logic_error( "Error - only 1,2,3D thread block shape allowed" );
    }

    SizeArray threadBlockShape( 3, 1 );
    std::copy( lwgs.begin(), lwgs.end(), threadBlockShape.begin() );
    threadBlockShape[ 2 ] = gwgs.size() == 3 ? gwgs[ 2 ] : 1; //force to be the same size as the 3rd dimension
                                       //of the global domain z-size
    SizeArray gridShape( 2, 1 );
    std::copy( gwgs.begin(), gwgs.end(), gridShape.begin() );
    gridShape[ 0 ] = ( gridShape[ 0 ] + threadBlockShape[ 0 ] - 1 ) / threadBlockShape[ 0 ];
    gridShape[ 1 ] = ( gridShape[ 1 ] + threadBlockShape[ 1 ] - 1 ) / threadBlockShape[ 1 ];

    return CUMemLayout( gridShape, threadBlockShape );
}


//------------------------------------------------------------------------------
CUMemLayout PrepareMemoryLayout( CUkernel k,
                                 const SizeArray& gwgs,
                                 const SizeArray& lwgs )
{
    
    
    CUMemLayout ml = ComputeCUDAMemLayout( gwgs, lwgs );
    CUresult status = ::cuFuncSetBlockShape( k,
                                             int( ml.threadBlockShape[ 0 ] ),
                                             int( ml.threadBlockShape[ 1 ] ),
                                             int( ml.threadBlockShape[ 2 ] ) );
    if( status != CUDA_SUCCESS )
    {
        throw std::runtime_error( "Error - cuFuncSetBlockShape() " + cuERRORS[ status ] );
    }
    return ml;
}

//------------------------------------------------------------------------------
CUMemLayout InvokeKernelAsync( CUkernel k,
                               const SizeArray& gwgs,
                               const SizeArray& lwgs,
                               const VArgList& valist )
{
    
    

    SetupKernelParameters( k, valist );
    CUMemLayout memLayout = PrepareMemoryLayout( k, gwgs, lwgs );
    CUresult status = ::cuLaunchGridAsync( k, int( memLayout.gridShape[ 0 ] ), int( memLayout.gridShape[ 1 ] ), 0 );
    if( status != CUDA_SUCCESS )
    {
        throw std::runtime_error( "Error - cuFuncSetBlockShape() " + cuERRORS[ status ] );
    }    
    return memLayout;
    
}    

//------------------------------------------------------------------------------
CUMemLayout InvokeKernelSync( CUkernel k,
                              const SizeArray& gwgs,
                              const SizeArray& lwgs,
                              const VArgList& valist )
{    
    SetupKernelParameters( k, valist );
    CUMemLayout memLayout = PrepareMemoryLayout( k, gwgs, lwgs );
    CUresult status = ::cuLaunchGrid( k, int( memLayout.gridShape[ 0 ] ), int( memLayout.gridShape[ 1 ] ) );
    if( status != CUDA_SUCCESS )
    {
        throw std::runtime_error( "Error - cuFuncSetBlockShape() " + cuERRORS[ status ] );
    }
    ::cuCtxSynchronize(); //WHICH CONTEXT ??
    return memLayout;
}    

//------------------------------------------------------------------------------
void ReleaseExecutionContext( CUDAExecutionContext& ec )
{
    ec.program.Release();
    ec.kernel.Release();
    ec.context.Release();
}
