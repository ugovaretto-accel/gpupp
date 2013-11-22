#ifndef ANY_H_
#define ANY_H_
//
// Copyright (c) 2010, 2011 - Ugo Varetto
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


/// @file Any.h Implementation of class to hold instances of any type.

#include <typeinfo>
#include <vector>
#include <iterator>


//------------------------------------------------------------------------------
/// @brief Class that can hold instances of any type. Uses defaule new/delete
/// operators.
/// @todo allow client code to specify allocator
/// @ingroup utility
class Any
{
public:
    /// Type used by Any::Type() method to signal an empty @c Any instance. 
    struct EMPTY_ {};
    /// Default constructor, sets the internal pointer to @c NULL
    Any() : pval_( 0 ) {}
    /// Constructor accepting a parameter copied into internal type instance.
    template < class ValT >
    Any( const ValT& v ) 
        : pval_( new ValHandler< ValT >( v ) )
    {} 
    /// Copy constructor.
    Any( const Any& a ) : pval_( a.pval_ ? a.pval_->Clone() : 0 ) {}
    /// Destructor: deletes the contained data type.
    ~Any() { delete pval_; }
public:
    /// Returns @c true if instance empty.
    bool Empty() const { return pval_ == 0; }
    /// Returns type of contained data or Any::EMPTY_ if instance empty.
    const std::type_info& Type() const
    { 
        //note gcc requires typeid(C) with C != void; compiles on vc++ 2008
        return !Empty() ? pval_->GetType() : typeid( EMPTY_ ); // 
    } 
    /// Swap two Any instances by swapping the internal pointers.
    Any& Swap( Any& a ) { std::swap( pval_, a.pval_ ); return *this; }
    /// Assignment
    Any& operator=( const Any& a ) { Any( a ).Swap( *this ); return *this; }
    /// Assignment from non - @c Any value.
    template < class ValT > 
    Any& operator=( const ValT& v )
    { 
        CheckAnyTypeAndThrow< ValT >( *this );
        Any( v ).Swap( *this ); return *this;
    }
    /// Equality: check by converting value to contained value type then
    /// invoking equality operator on converted type.
    template < class ValT > 
    bool operator==( const ValT& v ) const
    { 
        CheckAndThrow< ValT >(); 
        return ( static_cast< ValHandler<ValT>* >( pval_ )->val_ ) == v;
    }
public:
    ///Convert to const reference.
    template < class ValT >    operator const ValT&() const
    { 
        CheckAndThrow< ValT >();    
        return ( static_cast< ValHandler<ValT>* >( pval_ )->val_ );
    }
    ///Convert to reference.
    template < class ValT >    operator ValT&() const
    { 
        CheckAndThrow< ValT >();
        return ( static_cast< ValHandler<ValT>* >( pval_ )->val_ );
    }
    /// Check if contained data is convertible to specific type.
    /// @note although this function does not use any private data
    /// it is required to be defined withing the Any class since:
    /// - it cannot be defined before the class because it uses
    ///   Any's member functions
    /// - it cannot be defined after the class because in this case
    ///   other inline friends cannot invoke it
    template < class ValT > 
    friend void CheckAnyTypeAndThrow( const Any& any )
    {
//#ifdef ANY_CHECK_TYPE
        // note that virtual base members are required for
        // dynamic cast to work; use typeid functionality instead
        if( typeid( ValT ) != any.Type() )
            if(    !typeid( ValT ).before( any.Type() ) ) 
                throw std::logic_error( 
                        (std::string( " Attempt to convert from ") 
                        + any.Type().name()
                        + std::string( " to " )
                        + typeid( ValT ).name() ).c_str() );
//#endif
    }
    /// Return size of contained data.
    friend size_t AnySizeOf( const Any& any )
    {
        return any.pval_->SizeofData();
    }
    ///Give access to address of contained data.
    friend void* AnyAddress( Any& any )
    {
        return any.pval_->GetDataAddress();
    }
    ///Give access to address of contained data.
    friend size_t AnyAlignment( Any& any )
    {
        return any.pval_->GetAlignment();
    }
    ///Give access to address of contained data.
    friend const void* AnyAddress( const Any& any )
    {
        return any.pval_->GetDataAddress();
    }
    /// Give access to address of contained data.
    template < class AnyT >   
    friend AnyT* AnyPtr( Any& any )
    {
        CheckAnyTypeAndThrow< AnyT >( any );
        return ( &static_cast< Any::ValHandler<AnyT>* >( any.pval_ )->val_ );
    }
    /// Give access to address of contained const data. 
    template < class AnyT >   
    friend const AnyT* AnyPtr( const Any& any )
    {
        CheckAnyTypeAndThrow< AnyT >( any );
        return ( &static_cast< Any::ValHandler<AnyT>* >( any.pval_ )->val_ );
    }
    /// Give access to reference to contained data.
    template < class AnyT >   
    friend AnyT& AnyRef( Any& any )
    {
        CheckAnyTypeAndThrow< AnyT >( any );
        return ( static_cast< Any::ValHandler<AnyT>* >( any.pval_ )->val_ );
    }
    /// Give access to const reference to contained data.
    template < class AnyT >   
    friend const AnyT& AnyRef( const Any& any )
    {
        CheckAnyTypeAndThrow< AnyT >( any );
        return ( static_cast< Any::ValHandler<AnyT>* >( any.pval_ )->val_ );
    }
    /// Return value.
    template < class AnyT >   
    friend AnyT AnyVal( const Any& any )
    {
        CheckAnyTypeAndThrow< AnyT >( any );
        return ( static_cast< Any::ValHandler<AnyT>* >( any.pval_ )->val_ );
    }
    /// Proxy pointer
    struct ProxyPtr
    {
        Any& r;
        ProxyPtr( Any& any ) : r( any ) {}
        template < class D >
        operator D&()
        {
            return *AnyPtr< D >( r ); 
        }
        template < class D >
        operator const D&()
        {
            return *AnyPtr< D >( r ); 
        }
        template < class D >
        ProxyPtr operator=( const D& v )
        { 
            r = v;
            return ProxyPtr( r );
        }
    };

    /// Return a proxy pointer that points to the contained value.
    ProxyPtr operator*()
    {
        return ProxyPtr( *this );
    }

private:
    /// Check if contained data is convertible to specific type.
    template < class ValT > void CheckAndThrow() const
    {
        CheckAnyTypeAndThrow< ValT >( *this );
    }
    /// @interface HandlerBase Wrapper for data storage.
    struct HandlerBase // hint: use small object allocator
    {
        virtual const std::type_info& GetType() const = 0;
        virtual HandlerBase* Clone() const = 0;
        virtual ~HandlerBase() {}
        virtual size_t GetAlignment() const  = 0;
        virtual std::ostream& Serialize( std::ostream& os ) const = 0;
        virtual size_t SizeofData() const = 0;
        virtual void* GetDataAddress() = 0;
        virtual const void* GetDataAddress() const  = 0;
    };


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

    /// HandlerBase actual data container class.
    template < class T > struct ValHandler :  HandlerBase
    {
        typedef T Type;
        T val_;
        size_t alignment_;
        ValHandler( const T& v ) : val_( v ), alignment_( Align< T >::Value )
        {}
        const std::type_info& GetType() const { return typeid( T ); }
        ValHandler* Clone() const { return new ValHandler( val_ ); }
        std::ostream& Serialize( std::ostream& os ) const
        {
            os << val_;
            return os;
        }
        size_t SizeofData() const { return sizeof( Type ); }
        void* GetDataAddress() { return &val_; }
        const void* GetDataAddress() const { return &val_; }
        size_t GetAlignment() const { return alignment_; } 
    };

    ///Pointer to contained data: deleted when Any instance deleted.
    HandlerBase* pval_;    

    ///Overloaded operator to serialize data to output streams.
    friend inline std::ostream& operator<<( std::ostream& os, const Any& any )
    {
        if( any.Empty() ) return os;
        return any.pval_->Serialize( os );
    }

};

///Utility function to print the content of an std::vector of Any objects.
inline std::ostream& operator<<( std::ostream& os, const std::vector< Any >& av )
{
    std::copy( av.begin(), av.end(), std::ostream_iterator< Any >( os, ", " ) );
    return os;
}


#endif // ANY_H_

