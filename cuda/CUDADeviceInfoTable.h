#ifndef CUDA_DEVICE_INFO_TABLE_H_
#define CUDA_DEVICE_INFO_TABLE_H_
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

#include <map>
#include <string>
#include <cuda.h>


///Device info table. Bidirectional map: id -> string; string -> id
class CUDADeviceInfoTable
{
private:
    typedef std::map< CUdevice_attribute_enum, std::string > IdToString;
    typedef std::map< std::string, CUdevice_attribute_enum > StringToId;
public:    
    typedef IdToString::const_iterator IDIterator;
private:
    CUDADeviceInfoTable() 
    {
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK ] = "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK";    
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X ] = "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y ] = "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z ] = "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X ] = "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y ] = "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z ] = "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK ] = "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK"; 
        i2s_[ CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY ] = "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY";
        i2s_[ CU_DEVICE_ATTRIBUTE_WARP_SIZE ] = "CU_DEVICE_ATTRIBUTE_WARP_SIZE";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_PITCH ] = "CU_DEVICE_ATTRIBUTE_MAX_PITCH";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK ] = "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK"; 
        i2s_[ CU_DEVICE_ATTRIBUTE_CLOCK_RATE ] = "CU_DEVICE_ATTRIBUTE_CLOCK_RATE";
        i2s_[ CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT ] = "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT";
        i2s_[ CU_DEVICE_ATTRIBUTE_GPU_OVERLAP ] = "CU_DEVICE_ATTRIBUTE_GPU_OVERLAP";
        i2s_[ CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT ] = "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT";
        i2s_[ CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT ] = "CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT";
        i2s_[ CU_DEVICE_ATTRIBUTE_INTEGRATED ] = "CU_DEVICE_ATTRIBUTE_INTEGRATED";
        i2s_[ CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY ] = "CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY";
        i2s_[ CU_DEVICE_ATTRIBUTE_COMPUTE_MODE ] = "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH";
#if CUDA_VERSION >= 4000	
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH ]  = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH"; 
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT";
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS";
#endif        
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH"; 
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT";
        i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES";
        i2s_[ CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT ] = "CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT"; 
        i2s_[ CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS ] = "CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS"; 
        i2s_[ CU_DEVICE_ATTRIBUTE_ECC_ENABLED ] = "CU_DEVICE_ATTRIBUTE_ECC_ENABLED";
#if CUDA_VERSION >= 4000
		i2s_[ CU_DEVICE_ATTRIBUTE_PCI_BUS_ID ] = "CU_DEVICE_ATTRIBUTE_PCI_BUS_ID";                       
		i2s_[ CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID ] = "CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID";                 
		i2s_[ CU_DEVICE_ATTRIBUTE_TCC_DRIVER ] = "CU_DEVICE_ATTRIBUTE_TCC_DRIVER";                       
		i2s_[ CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE ] = "CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE";         
		i2s_[ CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH ] = "CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH";          
		i2s_[ CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE ] = "CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE";                     
		i2s_[ CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR ] = "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR";    
		i2s_[ CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT ] = "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT";               
		i2s_[ CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING ] = "CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING";                 
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH";   
		i2s_[ CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS ] = "CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS"; 
#endif
        for( IdToString::iterator i = i2s_.begin(); i != i2s_.end(); ++i )
        {
            s2i_[ i->second ] = i->first;
        }
    }
public:
    IdToString::mapped_type operator[]( IdToString::key_type k ) const 
    {
        if( i2s_.find( k ) == i2s_.end() ) throw std::range_error( "Device info value does not exist" );
        return i2s_.find( k )->second;
    }

    StringToId::mapped_type operator[]( StringToId::key_type k ) const 
    {
        if( s2i_.find( k ) == s2i_.end() ) throw std::range_error( "Device info value does not exist" );
        return s2i_.find( k )->second;
    }

    IDIterator DeviceIdBegin() const
    {
        return i2s_.begin();
    }

    IDIterator DeviceIdEnd() const
    {
        return i2s_.end();
    }
        
    static const CUDADeviceInfoTable& Instance()
    {
        static const CUDADeviceInfoTable i;
        return i;
    }

private:
    IdToString i2s_;
    StringToId s2i_;

};

#endif //CUDA_DEVICE_INFO_TABLE_H_