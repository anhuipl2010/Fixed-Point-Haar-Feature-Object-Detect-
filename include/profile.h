/*****************************************************************************
Copyright (c) 2009 - 2010 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 4169 $
$Date: 2010-11-12 21:18:29 +0530 (Fri, 12 Nov 2010) $

Title       : profile.h

Description : MACROs required for profiling

*****************************************************************************/
#ifndef __PROFILE_H__
#define __PROFILE_H__

/*=============  I N C L U D E S   =============*/
#include "adi_tool_chain.h"

/*==============  D E F I N E S  ===============*/

/* #define PROFILE */
#ifdef MISRA_RULES
#pragma diag(push)

/* MISRA Rule 19.10(Req): In the definition of a function-like macro
each instance of a parameter shall be enclosed in parentheses unless
it is used as the operand of # and ##.
*/
#pragma diag(suppress : misra_rule_19_10)

/*
MISRA Rule 19.4(Req): C macros shall only expand to a braced initialiser,
a constant, a string literal, a parenthesised expression, a type qualifier,
a storage class specifier, or a do-while-zero construct.
*/
#pragma diag(suppress : misra_rule_19_4)
#endif
#ifdef PROFILE
#define PROFBEG(TAG)    asm volatile("%0=cycles;" : "=d"(TAG.nCycles))
#define PROFEND(TAG)                                                   \
    asm volatile("r0=cycles; %0=r0-%0; %1=%0+%1;" : "+d"(TAG.nCycles), \
                 "+d"(TAG.nSum)::"R0")
#else
#define PROFBEG(TAG)
#define PROFEND(TAG)
#endif

/*=============  D A T A    T Y P E S   =============*/
typedef struct
{
    uint32_t    nCycles;
    uint32_t    nSum;
} prof_t;

/*=============  E X T E R N A L S  ============*/

/*
**  External Data section
*/
extern
prof_t  goProfile;
extern
prof_t  goProfileFunc;
extern
prof_t  goProfileFunc1;

/*
**  External Function Prototypes
*/
#ifdef MISRA_RULES
#pragma diag(pop)
#endif
#endif /* __PROFILE_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/tags/release_1_4_0_imagetoolbox_111510/demo/include/profile.h $
**
*/
