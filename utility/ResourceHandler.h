#ifndef RESOURCE_HANDLER_H_
#define RESOURCE_HANDLER_H_
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

#include <stdexcept>

//------------------------------------------------------------------------------
///Non-synchronized counter to keep track of number of references
class SimpleCounter
{
public:
    ///Constructor.
    /// \param c start value for counter
    SimpleCounter( unsigned c ) : count_( c ) {}
    ///Default constructor.
    SimpleCounter() : count_( 1 ) {}
    ///Increment counter.
    void Inc() {
        ++count_;
    }
    ///Decrement counter.
    /// \return value of decremented counter.
    unsigned Dec() {
        --count_;
        return count_;
    }
    ///Check if counter is zero.
    /// \return \c true if counter is zero
    bool Zero() const { return count_ == 0; }
    ///Return counter value
    /// \return current value of counter
    unsigned Count() const { return count_; } 
private:
    ///Counter.
    unsigned count_;
};


//------------------------------------------------------------------------------
///Generic resource handler for cases where the lifetime of resources is handled
///through functions
///\tparam ResourceT resource data type
///\tparam ReturnTypeT type of value returned by resource handling functions
///\tparam RetainFunT type of function that increases the resource reference count
///\tparam ReleaseFunT type of function used to release the resource reference count
///\tparam NameT printable name of resource; class needs to be convertible to const char*
///\tparam RETURN_SUCCESS_CODE identifier of successful operation
///\tparam CounterT reference counter
template< class ResourceT,
          class ReturnTypeT, 
// _WIN32 is always defined for 32 and 64 bit
#if defined( _WIN32 ) && !defined( _WIN64 )
          ReturnTypeT (__stdcall* RetainFunT)( ResourceT ),
          ReturnTypeT (__stdcall* ReleaseFunT)( ResourceT ),
#else
          ReturnTypeT RetainFunT( ResourceT ),
          ReturnTypeT ReleaseFunT( ResourceT ),
#endif
          class NameT,
          ReturnTypeT RETURN_SUCCESS_CODE,    
          class CounterT = SimpleCounter >
class ResourceHandler
{
public:
    ///Default constructor.
    ResourceHandler() : resource_( ResourceT() ), resourceName_( NameT() ), counter_( 0 ) {}
    ///Constructor.
    ///\param r resource handler
    ///\param count initial value of reference counter
    explicit ResourceHandler( ResourceT r, unsigned count = 1 ) : resource_( r ), resourceName_( NameT() ),
        counter_( new CounterT( count ) ) {}
    ///Copy constructor
    ResourceHandler( const ResourceHandler& r ) : resource_( r.resource_ ), resourceName_( NameT() ),
        counter_( 0 )
    {
        AcquireResource( r );        
    }
    ///Assignment operator.
    ///\param r other resource handler instance
    ResourceHandler operator=( const ResourceHandler& r )
    {
        if( &r == this ) return *this;
        ReleaseResource();
        AcquireResource( r );
        return *this;
    }
    ///Automatic conversion to resource handler type.
    operator ResourceT() const { return resource_; }
    ///Returns resource handler.
    ResourceT Handle() const { return resource_; }
    ///Returns resource readable name; works only if the tamplate parameter NameT can be converted to a const char*.
    const char* Name() const { return resourceName_; }
    ///Returns current reference count of resource.
    unsigned RefCount() const { return counter_->Count(); }
    ///Release resource.
    void Release() { ReleaseResource(); }
    ///Release resource when destroyed.
    ~ResourceHandler()
    {
        ReleaseResource();
    }
private:
    ///Acquire resource from other handler.
    void AcquireResource( const ResourceHandler& rh )
    {
        if( rh.counter_ == 0 ) return;
        counter_ = rh.counter_;
        resource_ = rh.resource_;
        counter_->Inc();
    }
    ///Release resource.
    ///\throw std::runtime_error in case an error is returned by the release function
    void ReleaseResource()
    {
        if( !counter_ ) return;
        if( counter_->Dec() == 0 )
        {
            if( ReleaseFunT( resource_ ) != ReturnTypeT( RETURN_SUCCESS_CODE ) )
            {
                throw( std::runtime_error( "Error: releasing resource \"" + resourceName_ + "\"" ) );
            }
            delete counter_;
            counter_ = 0;
        }                
    }
private:
    ///Resource handler
    ResourceT resource_;
    ///Resource name.
    std::string resourceName_;
    //Counter instance.
    CounterT* counter_;
};

#endif //RESOURCE_HANDLER_H_