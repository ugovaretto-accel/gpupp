#
# Copyright (c) 2010 - Ugo Varetto
#
# This source code is free; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This source code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this source code; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA  02110-1301, USA.
# 

cmake_minimum_required(VERSION 2.6)

project(gpupp-test)

#Automatic discovery supported for NVIDIA only at this time
set(OPENCL_DEF_INCLUDE_DIR "$ENV{CUDA_INC_PATH}")
set(OPENCL_DEF_LINK_DIR "$ENV{CUDA_LIB_PATH}")

set(OPENCL_INCLUDE_DIR ${OPENCL_DEF_INCLUDE_DIR} CACHE PATH "OpenCL include dir" )
set(OPENCL_LINK_DIR ${OPENCL_DEF_LINK_DIR} CACHE PATH "OpenCL lib dir" )

include_directories( ${OPENCL_INCLUDE_DIR} )
link_directories( ${OPENCL_LINK_DIR} )

set( COMMON_SRCS utility/ResourceHandler.h utility/Any.h utility/varargs.h utility/CmdLine.h utility/Timer.h )
set( OPENCL_SRCS opencl/gpupp.cpp opencl/gpupp.h opencl/OpenCLDeviceInfoTable.h opencl/OpenCLStatusCodesTable.h )
set( TEST_CL_SRCS  gpupp-test-cl.cpp )
set( CUDA_SRCS utility/alignment.h cuda/gpupp.cpp cuda/gpupp.h cuda/CUDADeviceInfoTable.h cuda/CUDAStatusCodesTable.h )
set( TEST_CUDA_SRCS  gpupp-test-cu.cpp )

include(FindCUDA)
include_directories( ${CUDA_INCLUDE_DIRS} )

add_executable( gpupp-test-cl ${OPENCL_SRCS} ${COMMON_SRCS} ${TEST_CL_SRCS} )
add_executable( gpupp-test-cu ${CUDA_SRCS} ${COMMON_SRCS} ${TEST_CUDA_SRCS} )

set(CLLIB OpenCL)
target_link_libraries( gpupp-test-cl ${CLLIB} )
target_link_libraries( gpupp-test-cu ${CUDA_LIBRARIES} )