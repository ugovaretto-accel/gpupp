#ifndef VARARGS_H_
#define VARARGS_H_
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
#include <list>
#include <iostream>

#include "Any.h"

///Variable argument list: contains a sequence of instances of Any type;
///can store values of any type and it is intended to be used to call
///functions with a variable number of arguments through the overloaded
/// operator ',' e.g.:
///\code
///  InvokeKernelSync( ec, globalWGroupSize, localWGroupSize,
///                   ( VArgList(),  //<- Marks the beginning of a variable argument list
///                     inMatD,
///                     MATRIX_WIDTH ,
///                     MATRIX_HEIGHT ,
///                     inVecD,
///                     outVecD
///                    )              //<- Marks the end of a variable argument list
///                  );
///\endcode 
class VArgList
{
public:
    typedef std::list< Any > ArgsType;
    typedef ArgsType::size_type size_type;
    typedef ArgsType::iterator ArgListIterator;
    typedef ArgsType::const_iterator ArgListConstIterator;
    ///Constructor. Constructs a list from a single parameter.
    ///\param a parameter added to list.
    VArgList( const Any& a ) { l_.push_back( a ); }
    ///Default constructor.
    VArgList() {}
    ///Returns iterator pointing at beginning of parameter sequence.
    ArgListConstIterator Begin() const { return l_.begin(); }
    ///Returns iterator pointing at the end of parameter sequence.
    ArgListConstIterator End() const { return l_.end(); }
    ///Returns a constant iterator pointing at beginning of parameter sequence.
    ArgListIterator Begin() { return l_.begin(); }
    ///Returns a constant iterator pointing at end of parameter sequence.
    ArgListIterator End() { return l_.end(); }
    ///Returns number of values in list.
    size_type Size() const { return l_.size(); }
public:
    ///Overloaded \c ',' operator.
    ///\param al argument list
    ///\param a parameter to be added to list
    friend VArgList operator,( VArgList al, const Any& a )
    {
        al.l_.push_back( a );
        return al;
    }
private:
    ArgsType l_;
};

#endif //VARARGS_H_