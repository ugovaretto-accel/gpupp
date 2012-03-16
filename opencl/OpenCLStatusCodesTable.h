#ifndef OPENCL_STATUS_CODES_TABLE_H_
#define OPENCL_STATUS_CODES_TABLE_H_
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


///Status/Error code table. Bidirectional map: id -> string; string -> id
class OpenCLStatusCodesTable
{
private:
    typedef std::map< int, std::string > IdToString;
    typedef std::map< std::string, int > StringToId;
public:    
    typedef IdToString::const_iterator IDIterator;
private:
    OpenCLStatusCodesTable() 
    {
        i2s_[ CL_SUCCESS                          ] = "CL_SUCCESS";                        
        i2s_[ CL_DEVICE_NOT_FOUND                 ] = "CL_DEVICE_NOT_FOUND";                
        i2s_[ CL_DEVICE_NOT_AVAILABLE             ] = "CL_DEVICE_NOT_AVAILABLE";           
        i2s_[ CL_COMPILER_NOT_AVAILABLE           ] = "CL_COMPILER_NOT_AVAILABLE";          
        i2s_[ CL_MEM_OBJECT_ALLOCATION_FAILURE    ] = "CL_MEM_OBJECT_ALLOCATION_FAILURE";   
        i2s_[ CL_OUT_OF_RESOURCES                 ] = "CL_OUT_OF_RESOURCES";                
        i2s_[ CL_OUT_OF_HOST_MEMORY               ] = "CL_OUT_OF_HOST_MEMORY";              
        i2s_[ CL_PROFILING_INFO_NOT_AVAILABLE     ] = "CL_PROFILING_INFO_NOT_AVAILABLE";    
        i2s_[ CL_MEM_COPY_OVERLAP                 ] = "CL_MEM_COPY_OVERLAP";                
        i2s_[ CL_IMAGE_FORMAT_MISMATCH            ] = "CL_IMAGE_FORMAT_MISMATCH";   
        i2s_[ CL_IMAGE_FORMAT_NOT_SUPPORTED       ] = "CL_IMAGE_FORMAT_NOT_SUPPORTED";     
        i2s_[ CL_BUILD_PROGRAM_FAILURE            ] = "CL_BUILD_PROGRAM_FAILURE";          
        i2s_[ CL_MAP_FAILURE                      ] = "CL_MAP_FAILURE";
        i2s_[ CL_INVALID_VALUE                    ] = "CL_INVALID_VALUE";               
        i2s_[ CL_INVALID_DEVICE_TYPE              ] = "CL_INVALID_DEVICE_TYPE ";           
        i2s_[ CL_INVALID_PLATFORM                 ] = "CL_INVALID_PLATFORM";               
        i2s_[ CL_INVALID_DEVICE                   ] = "CL_INVALID_DEVICE";                
        i2s_[ CL_INVALID_CONTEXT                  ] = "CL_INVALID_CONTEXT";                
        i2s_[ CL_INVALID_QUEUE_PROPERTIES         ] = "CL_INVALID_QUEUE_PROPERTIES";       
        i2s_[ CL_INVALID_COMMAND_QUEUE            ] = "CL_INVALID_COMMAND_QUEUE";          
        i2s_[ CL_INVALID_HOST_PTR                 ] = "CL_INVALID_HOST_PTR";               
        i2s_[ CL_INVALID_MEM_OBJECT               ] = "CL_INVALID_MEM_OBJECT";             
        i2s_[ CL_INVALID_IMAGE_FORMAT_DESCRIPTOR  ] = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";   
        i2s_[ CL_INVALID_IMAGE_SIZE               ] = "CL_INVALID_IMAGE_SIZE";
        i2s_[ CL_INVALID_SAMPLER                  ] = "CL_INVALID_SAMPLER";                
        i2s_[ CL_INVALID_BINARY                   ] = "CL_INVALID_BINARY";                 
        i2s_[ CL_INVALID_BUILD_OPTIONS            ] = "CL_INVALID_BUILD_OPTIONS";          
        i2s_[ CL_INVALID_PROGRAM                  ] = "CL_INVALID_PROGRAM";                
        i2s_[ CL_INVALID_PROGRAM_EXECUTABLE       ] = "CL_INVALID_PROGRAM_EXECUTABLE";     
        i2s_[ CL_INVALID_KERNEL_NAME              ] = "CL_INVALID_KERNEL_NAME";            
        i2s_[ CL_INVALID_KERNEL_DEFINITION        ] = "CL_INVALID_KERNEL_DEFINITION";      
        i2s_[ CL_INVALID_KERNEL                   ] = "CL_INVALID_KERNEL";                 
        i2s_[ CL_INVALID_ARG_INDEX                ] = "CL_INVALID_ARG_INDEX";              
        i2s_[ CL_INVALID_ARG_VALUE                ] = "CL_INVALID_ARG_VALUE";              
        i2s_[ CL_INVALID_ARG_SIZE                 ] = "CL_INVALID_ARG_SIZE";               
        i2s_[ CL_INVALID_KERNEL_ARGS              ] = "CL_INVALID_KERNEL_ARGS";            
        i2s_[ CL_INVALID_WORK_DIMENSION           ] = "CL_INVALID_WORK_DIMENSION";         
        i2s_[ CL_INVALID_WORK_GROUP_SIZE          ] = "CL_INVALID_WORK_GROUP_SIZE";        
        i2s_[ CL_INVALID_WORK_ITEM_SIZE           ] = "CL_INVALID_WORK_ITEM_SIZE";         
        i2s_[ CL_INVALID_GLOBAL_OFFSET            ] = "CL_INVALID_GLOBAL_OFFSET";          
        i2s_[ CL_INVALID_EVENT_WAIT_LIST          ] = "CL_INVALID_EVENT_WAIT_LIST";        
        i2s_[ CL_INVALID_EVENT                    ] = "CL_INVALID_EVENT";                  
        i2s_[ CL_INVALID_OPERATION                ] = "CL_INVALID_OPERATION";              
        i2s_[ CL_INVALID_GL_OBJECT                ] = "CL_INVALID_GL_OBJECT";              
        i2s_[ CL_INVALID_BUFFER_SIZE              ] = "CL_INVALID_BUFFER_SIZE";            
        i2s_[ CL_INVALID_MIP_LEVEL                ] = "CL_INVALID_MIP_LEVEL";              
        i2s_[ CL_INVALID_GLOBAL_WORK_SIZE         ] = "CL_INVALID_GLOBAL_WORK_SIZE";       
        for( IdToString::iterator i = i2s_.begin(); i != i2s_.end(); ++i )
        {
            s2i_[ i->second ] = i->first;
        }
    }
public:
    IdToString::mapped_type operator[]( IdToString::key_type k ) const 
    {
        if( i2s_.find( k ) == i2s_.end() ) throw std::range_error( "Unknown status code" );
        return i2s_.find( k )->second;
    }

    StringToId::mapped_type operator[]( StringToId::key_type k ) const 
    {
        if( s2i_.find( k ) == s2i_.end() ) throw std::range_error( "Unknowns status code" );
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
        
    static const OpenCLStatusCodesTable& Instance()
    {
        static const OpenCLStatusCodesTable i;
        return i;
    }

private:
    IdToString i2s_;
    StringToId s2i_;

};

#endif //OPENCL_STATUS_CODES_TABLE_H_