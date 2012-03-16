#include <iostream>
#include <utility>

#include "utility/CmdLine.h"

//------------------------------------------------------------------------------
//test with: program --first-command hey --second-command -tc 4 7.2 12
void TestCmdLine( int argc, char** argv )
{
    const bool DO_NOT_REPORT_UNKNOWN_PARAMETERS = false;
    const bool OPTIONAL_COMMAND = true;
    const bool REQUIRED_COMMAND = !OPTIONAL_COMMAND;
    CmdLine cmd( DO_NOT_REPORT_UNKNOWN_PARAMETERS );
    cmd.Add( "my first mandatory command with zero or one arguments", "first-command", "fc", std::make_pair( 0, 1 ), REQUIRED_COMMAND );
    cmd.Add( "my second optional command without arguments", "second-command", "sc", std::make_pair( 0, 0 ), OPTIONAL_COMMAND );
    cmd.Add( "my third optional command with 3 arguments", "third-command", "tc", std::make_pair( 3, 3 ), OPTIONAL_COMMAND );
    try
    {
        // IMPORTANT: NEVER PASS argc AND argv directly WHEN reportUnknownParameters IS SET TO TRUE
        // since it will block when parsing parameter[0]; use --argc, ++argv in this case
        CmdLine::ParsedEntries pe = cmd.ParseCommandLine( argc, argv ); 
        const double p = Get< double >( pe[ "third-command" ][ 1 ] );
        std::cout << "second argument of -tc command: " << p <<  std::endl;
        std::cout << cmd.HelpText() << std::endl;
    }
    catch( const std::exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{    
   TestCmdLine( argc, argv );
#ifdef _MSC_VER
#ifdef _DEBUG
    std::cout << "\n<press Enter to exit>" << std::endl;
    ::getchar();
#endif
#endif
    return 0;
}

