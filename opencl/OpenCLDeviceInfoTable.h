#ifndef OPENCL_DEVICE_INFO_TABLE_H_
#define OPENCL_DEVICE_INFO_TABLE_H_
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
#include <CL/cl.h>


///Device info table. Bidirectional map: id -> string; string -> id
class OpenCLDeviceInfoTable
{
private:
    typedef std::map< int, std::string > IdToString;
    typedef std::map< std::string, int > StringToId;
public:    
    typedef IdToString::const_iterator IDIterator;
private:
    OpenCLDeviceInfoTable() //52 parameters!
    {
        //unsigned ints
        i2s_[ CL_DEVICE_TYPE                          ] = "CL_DEVICE_TYPE";
        i2s_[ CL_DEVICE_VENDOR_ID                     ] = "CL_DEVICE_VENDOR_ID";
        i2s_[ CL_DEVICE_MAX_COMPUTE_UNITS             ] = "CL_DEVICE_MAX_COMPUTE_UNITS";
        i2s_[ CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS      ] = "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS";
        i2s_[ CL_DEVICE_MAX_WORK_GROUP_SIZE           ] = "CL_DEVICE_MAX_WORK_GROUP_SIZE";
        i2s_[ CL_DEVICE_MAX_WORK_ITEM_SIZES           ] = "CL_DEVICE_MAX_WORK_ITEM_SIZES";
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR   ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR";
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT";
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT    ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT";
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG   ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG";
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT"; 
        i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE ] = "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE";  
        i2s_[ CL_DEVICE_MAX_CLOCK_FREQUENCY           ] = "CL_DEVICE_MAX_CLOCK_FREQUENCY";
        i2s_[ CL_DEVICE_ADDRESS_BITS                  ] = "CL_DEVICE_ADDRESS_BITS";
        i2s_[ CL_DEVICE_MAX_READ_IMAGE_ARGS           ] = "CL_DEVICE_MAX_READ_IMAGE_ARGS";
        i2s_[ CL_DEVICE_MAX_WRITE_IMAGE_ARGS          ] = "CL_DEVICE_MAX_WRITE_IMAGE_ARGS";    
        i2s_[ CL_DEVICE_MAX_MEM_ALLOC_SIZE            ] = "CL_DEVICE_MAX_MEM_ALLOC_SIZE";
        i2s_[ CL_DEVICE_IMAGE2D_MAX_WIDTH             ] = "CL_DEVICE_IMAGE2D_MAX_WIDTH";
        i2s_[ CL_DEVICE_IMAGE2D_MAX_HEIGHT            ] = "CL_DEVICE_IMAGE2D_MAX_HEIGHT";
        i2s_[ CL_DEVICE_IMAGE3D_MAX_WIDTH             ] = "CL_DEVICE_IMAGE3D_MAX_WIDTH";
        i2s_[ CL_DEVICE_IMAGE3D_MAX_HEIGHT            ] = "CL_DEVICE_IMAGE3D_MAX_HEIGHT";
        i2s_[ CL_DEVICE_IMAGE3D_MAX_DEPTH             ] = "CL_DEVICE_IMAGE3D_MAX_DEPTH";
        i2s_[ CL_DEVICE_IMAGE_SUPPORT                 ] = "CL_DEVICE_IMAGE_SUPPORT";
        i2s_[ CL_DEVICE_MAX_PARAMETER_SIZE            ] = "CL_DEVICE_MAX_PARAMETER_SIZE";
        i2s_[ CL_DEVICE_MAX_SAMPLERS                  ] = "CL_DEVICE_MAX_SAMPLERS";
        i2s_[ CL_DEVICE_MEM_BASE_ADDR_ALIGN           ] = "CL_DEVICE_MEM_BASE_ADDR_ALIGN";
        i2s_[ CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE      ] = "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE";
        i2s_[ CL_DEVICE_SINGLE_FP_CONFIG              ] = "CL_DEVICE_SINGLE_FP_CONFIG";
        i2s_[ CL_DEVICE_GLOBAL_MEM_CACHE_TYPE         ] = "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE";
        i2s_[ CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE     ] = "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE";
        i2s_[ CL_DEVICE_GLOBAL_MEM_CACHE_SIZE         ] = "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE";
        i2s_[ CL_DEVICE_GLOBAL_MEM_SIZE               ] = "CL_DEVICE_GLOBAL_MEM_SIZE";
        i2s_[ CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE      ] = "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE";
        i2s_[ CL_DEVICE_MAX_CONSTANT_ARGS             ] = "CL_DEVICE_MAX_CONSTANT_ARGS";
        i2s_[ CL_DEVICE_LOCAL_MEM_TYPE                ] = "CL_DEVICE_LOCAL_MEM_TYPE";
        i2s_[ CL_DEVICE_LOCAL_MEM_SIZE                ] = "CL_DEVICE_LOCAL_MEM_SIZE";
        i2s_[ CL_DEVICE_ERROR_CORRECTION_SUPPORT      ] = "CL_DEVICE_ERROR_CORRECTION_SUPPORT";    
        i2s_[ CL_DEVICE_PROFILING_TIMER_RESOLUTION    ] = "CL_DEVICE_PROFILING_TIMER_RESOLUTION";
        i2s_[ CL_DEVICE_ENDIAN_LITTLE                 ] = "CL_DEVICE_ENDIAN_LITTLE";
        i2s_[ CL_DEVICE_AVAILABLE                     ] = "CL_DEVICE_AVAILABLE";
        i2s_[ CL_DEVICE_COMPILER_AVAILABLE            ] = "CL_DEVICE_COMPILER_AVAILABLE";
        i2s_[ CL_DEVICE_EXECUTION_CAPABILITIES        ] = "CL_DEVICE_EXECUTION_CAPABILITIES";
        i2s_[ CL_DEVICE_QUEUE_PROPERTIES              ] = "CL_DEVICE_QUEUE_PROPERTIES";
        //strings
        i2s_[ CL_DEVICE_NAME                          ] = "CL_DEVICE_NAME";
        i2s_[ CL_DEVICE_VENDOR                        ] = "CL_DEVICE_VENDOR";
        i2s_[ CL_DRIVER_VERSION                       ] = "CL_DRIVER_VERSION";
        i2s_[ CL_DEVICE_PROFILE                       ] = "CL_DEVICE_PROFILE";
        i2s_[ CL_DEVICE_VERSION                       ] = "CL_DEVICE_VERSION";
        i2s_[ CL_DEVICE_EXTENSIONS                    ] = "CL_DEVICE_EXTENSIONS";
        i2s_[ CL_DEVICE_PLATFORM                      ] = "CL_DEVICE_PLATFORM";
        // 0x1032 reserved for CL_DEVICE_DOUBLE_FP_CONFIG
        // 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG
#ifdef CL_VERSION_1_1
		i2s_[ CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF ] =  "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF";      
		i2s_[ CL_DEVICE_HOST_UNIFIED_MEMORY ] = "CL_DEVICE_HOST_UNIFIED_MEMORY";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_INT ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE";
		i2s_[ CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF ] = "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF";
		i2s_[ CL_DEVICE_OPENCL_C_VERSION ] = "CL_DEVICE_OPENCL_C_VERSION";
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
        
    static const OpenCLDeviceInfoTable& Instance()
    {
        static const OpenCLDeviceInfoTable i;
        return i;
    }

    bool IsUInt( int id ) const 
    {
        return id < CL_DEVICE_NAME  || id > CL_DEVICE_EXTENSIONS;
    }

private:
    IdToString i2s_;
    StringToId s2i_;

};

#endif //OPENCL_DEVICE_INFO_TABLE_H_