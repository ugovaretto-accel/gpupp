#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_
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

/// Struct for computing alignment of data.
template < typename T > struct Align
{
    struct D
    {
        T d;
        char c[ 1 ];
    };
    enum {Value = sizeof( D ) - sizeof( T )};
};


/// Computes the alignment of a type
#define ALIGNMENT( T ) Align< T >::Value 

/// Computes the alignment of a value
template < typename T > 
inline int Alignment( const T& t ) { return ALIGNMENT( T ); }

/// Computes a new offset aligned according to the passes alignment value.
/// Templated to support any integer type
template < typename T >
inline T AlignedOffset( T off, int align )
{
    T m = off % align;
    return m != 0 ? off - off % align + align : off;
}

#endif //ALIGNMENT_H_