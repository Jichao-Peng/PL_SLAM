/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2016, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#ifndef gui_link_pragmas_H
#define gui_link_pragmas_H

#include <mrpt/config.h>
#include <mrpt/utils/boost_join.h>

// ** Important! **
// In each mrpt library, search and replace:
//  MRPT_XXX_EXPORT, MRPT_XXX_IMPORT
//  GUI_IMPEXP, mrpt_xxx_EXPORTS

/*   The macros below for DLL import/export are required for Windows only.
    Mostly all the definitions in this file are copied or at least guid
     on the file wx/dlimpexp.h, written by Vadim Zeitlin and published
	 under the wxWindows licence.
*/
#if defined(MRPT_OS_WINDOWS)
    /*
       __declspec works in BC++ 5 and later, Watcom C++ 11.0 and later as well
       as VC++ and gcc
     */
#    if defined(_MSC_VER) || defined(__BORLANDC__) || defined(__GNUC__) || defined(__WATCOMC__)
#        define MRPT_GUI_EXPORT __declspec(dllexport)
#        define MRPT_GUI_IMPORT __declspec(dllimport)
#    else /* compiler doesn't support __declspec() */
#        define MRPT_GUI_EXPORT
#        define MRPT_GUI_IMPORT
#    endif
#elif defined(MRPT_OS_OS2)		/* was __WXPM__ */
#    if defined (__WATCOMC__)
#        define MRPT_GUI_EXPORT __declspec(dllexport)
        /*
           __declspec(dllimport) prepends __imp to imported symbols. We do NOT
           want that!
         */
#        define MRPT_GUI_IMPORT
#    elif defined(__EMX__)
#        define MRPT_GUI_EXPORT
#        define MRPT_GUI_IMPORT
#    elif (!(defined(__VISAGECPP__) && (__IBMCPP__ < 400 || __IBMC__ < 400 )))
#        define MRPT_GUI_EXPORT _Export
#        define MRPT_GUI_IMPORT _Export
#    endif
#elif defined(MRPT_OS_APPLE)
#    ifdef __MWERKS__
#        define MRPT_GUI_EXPORT __declspec(export)
#        define MRPT_GUI_IMPORT __declspec(import)
#    endif
#elif defined(__CYGWIN__)
#    define MRPT_GUI_EXPORT __declspec(dllexport)
#    define MRPT_GUI_IMPORT __declspec(dllimport)
#endif

/* for other platforms/compilers we don't anything */
#ifndef MRPT_GUI_EXPORT
#    define MRPT_GUI_EXPORT
#    define MRPT_GUI_IMPORT
#endif

/*  Macros that map to export declaration when building the DLL, to import
	declaration if using it or to nothing at all if we are not compiling as DLL */
#if defined(MRPT_BUILT_AS_DLL)
#	if defined(mrpt_gui_EXPORTS)  /* Building the DLL */
#		define GUI_IMPEXP MRPT_GUI_EXPORT
#	else  /* Using the DLL */
#		define GUI_IMPEXP MRPT_GUI_IMPORT
#	endif
#else /* not making nor using DLL */
#    define GUI_IMPEXP
#endif


#endif
