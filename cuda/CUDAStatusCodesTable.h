#ifndef CUDA_STATUS_CODES_TABLE_H_
#define CUDA_STATUS_CODES_TABLE_H_
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


///Status/Error code table. Bidirectional map: id -> string; string -> id
class CUDAStatusCodesTable
{
private:
    typedef std::map< cudaError_enum, std::string > IdToString;
    typedef std::map< std::string, cudaError_enum > StringToId;
public:    
    typedef IdToString::const_iterator IDIterator;
private:
    CUDAStatusCodesTable() 
    {
        i2s_[ CUDA_SUCCESS                              ] = "CUDA_SUCCESS";                        
        i2s_[ CUDA_ERROR_INVALID_VALUE                  ] = "CUDA_ERROR_INVALID_VALUE";                
        i2s_[ CUDA_ERROR_OUT_OF_MEMORY                  ] = "CUDA_ERROR_OUT_OF_MEMORY";           
        i2s_[ CUDA_ERROR_NOT_INITIALIZED                ] = "CUDA_ERROR_NOT_INITIALIZED";          
        i2s_[ CUDA_ERROR_DEINITIALIZED                  ] = "CUDA_ERROR_DEINITIALIZED";   
        i2s_[ CUDA_ERROR_NO_DEVICE                      ] = "CUDA_ERROR_NO_DEVICE";              
        i2s_[ CUDA_ERROR_INVALID_DEVICE                 ] = "CUDA_ERROR_INVALID_DEVICE";    
        i2s_[ CUDA_ERROR_INVALID_IMAGE                  ] = "CUDA_ERROR_INVALID_IMAGE";   
        i2s_[ CUDA_ERROR_INVALID_CONTEXT                ] = "CUDA_ERROR_INVALID_CONTEXT";     
        i2s_[ CUDA_ERROR_CONTEXT_ALREADY_CURRENT        ] = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";          
        i2s_[ CUDA_ERROR_MAP_FAILED                     ] = "CUDA_ERROR_MAP_FAILED";
        i2s_[ CUDA_ERROR_UNMAP_FAILED                   ] = "CUDA_ERROR_UNMAP_FAILED";               
        i2s_[ CUDA_ERROR_ARRAY_IS_MAPPED                ] = "CUDA_ERROR_ARRAY_IS_MAPPED";           
        i2s_[ CUDA_ERROR_ALREADY_MAPPED                 ] = "CUDA_ERROR_ALREADY_MAPPED";               
        i2s_[ CUDA_ERROR_NO_BINARY_FOR_GPU              ] = "CUDA_ERROR_NO_BINARY_FOR_GPU";                
        i2s_[ CUDA_ERROR_ALREADY_ACQUIRED               ] = "CUDA_ERROR_ALREADY_ACQUIRED";                
        i2s_[ CUDA_ERROR_NOT_MAPPED                     ] = "CUDA_ERROR_NOT_MAPPED";       
        i2s_[ CUDA_ERROR_NOT_MAPPED_AS_ARRAY            ] = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";          
        i2s_[ CUDA_ERROR_NOT_MAPPED_AS_POINTER          ] = "CUDA_ERROR_NOT_MAPPED_AS_POINTER";               
        i2s_[ CUDA_ERROR_ECC_UNCORRECTABLE              ] = "CUDA_ERROR_ECC_UNCORRECTABLE";  
        i2s_[ CUDA_ERROR_INVALID_SOURCE                 ] = "CUDA_ERROR_INVALID_SOURCE";
        i2s_[ CUDA_ERROR_FILE_NOT_FOUND                 ] = "CUDA_ERROR_FILE_NOT_FOUND";
        i2s_[ CUDA_ERROR_INVALID_HANDLE                 ] = "CUDA_ERROR_INVALID_HANDLE";  
        i2s_[ CUDA_ERROR_NOT_FOUND                      ] = "CUDA_ERROR_NOT_FOUND"; 
        i2s_[ CUDA_ERROR_NOT_READY                      ] = "CUDA_ERROR_NOT_READY";
        i2s_[ CUDA_ERROR_LAUNCH_FAILED                  ] = "CUDA_ERROR_LAUNCH_FAILED";              
        i2s_[ CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        ] = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";              
        i2s_[ CUDA_ERROR_LAUNCH_TIMEOUT                 ] = "CUDA_ERROR_LAUNCH_TIMEOUT";               
        i2s_[ CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  ] = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
		// NOT AVAILABLE ANYMORE IN CUDA 4.0
        //i2s_[ CUDA_ERROR_POINTER_IS_64BIT               ] = "CUDA_ERROR_POINTER_IS_64BIT";        
        //i2s_[ CUDA_ERROR_SIZE_IS_64BIT                  ] = "CUDA_ERROR_SIZE_IS_64BIT";  
        i2s_[ CUDA_ERROR_UNKNOWN                        ] = "CUDA_ERROR_UNKNOWN";        
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
        
    static const CUDAStatusCodesTable& Instance()
    {
        static const CUDAStatusCodesTable i;
        return i;
    }

private:
    IdToString i2s_;
    StringToId s2i_;

};

#endif //CUDA_STATUS_CODES_TABLE_H_