/*****************************************************************************
Copyright (c) 2009 - 2010 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 3953 $
$Date: 2010-10-29 15:18:31 +0530 (Fri, 29 Oct 2010) $

Title       : adi_tool_chain.h

Description : contains structures and defines that are applicable/useful
              for all ADI codecs

*****************************************************************************/
#ifndef __ADI_TOOL_CHAIN_H__
#define __ADI_TOOL_CHAIN_H__

/*=============  I N C L U D E S   =============*/
#if defined _LANGUAGE_C
#ifdef __VISUALDSPVERSION__
#include <adi_types.h>
#endif /* __VISUALDSPVERSION__ */
#endif /*_LANGUAGE_C  */

/*=============  D E F I N E S   =============*/
#ifdef MISRA_RULES
#pragma diag(push)

/*
MISRA Rule 19.4(Req): C macros shall only expand to a braced initialiser, a
constant, a string literal, a parenthesised expression, a type
qualifier, a storage class specifier, or a do-while-zero construct.
*/
#pragma diag(suppress : misra_rule_19_4)

/*
MISRA Rule 19.10(Req): In the definition of a function-like macro each
instance of a parameter shall be enclosed in parentheses unless it is
used as the operand of # and ##.
*/
#pragma diag(suppress : misra_rule_19_10)

/*
MISRA Rule 19.12(Req): There shall be at most one occurrence of the # or ##
preprocessor operators in a single macro definition.
*/
#pragma diag(suppress : misra_rule_19_12)

/*
MISRA Rule 19.7(Adv): A function shall be used in preference to
a function-like macro.
*/
#pragma diag(suppress : misra_rule_19_7)

/*
MISRA Rule 19.13(Adv): The # and ## preprocessor operators should not be used.
*/
#pragma diag(suppress : misra_rule_19_13)
#endif
#ifdef _LANGUAGE_ASM
#ifdef __VISUALDSPVERSION__
#define __BYTE4 .byte4 =
#define __BYTE2 .byte2 =
#define __BYTE  .byte =
#else
#define __BYTE4 .byte4
#define __BYTE2 .byte2
#define __BYTE  .byte
#endif
#endif

/*
** Add more ADI standard section names to the below
** list upon requirement/usage
*/
#ifndef __VISUALDSPVERSION__
#ifndef _SHARED_LIB
#ifdef _LANGUAGE_ASM
#define adi_fast_prio0_r        .l1.data
#define adi_fastb0_prio0_r      .l1.data.a
#define adi_fastb0_prio0_rw     .l1.data.a
#define adi_fastb0_prio0_temp   .l1.data.a
#define adi_fastb1_prio0_r      .l1.data.b
#define adi_fastb1_prio0_rw     .l1.data.b
#define adi_fastb1_prio0_temp   .l1.data.b
#define adi_slow_noprio_r       .data
#define adi_fast_prio0_code     .l1.text
#define adi_slow_noprio_code    .text
#else /* _LANGUAGE_ASM  */
#define adi_fast_prio0_r        l1_data
#define adi_fast_prio0_rw       l1_data
#define adi_fast_prio0_temp     l1_data
#define adi_fast_prio1_r        l1_data
#define adi_fastb0_prio0_r      l1_data_A
#define adi_fastb0_prio0_rw     l1_data_A
#define adi_fastb0_prio0_temp   l1_data_A
#define adi_fastb1_prio0_r      l1_data_B
#define adi_fastb1_prio0_rw     l1_data_B
#define adi_fastb1_prio1_r      l1_data_B
#define adi_fastb1_prio0_temp   l1_data_B
#define adi_slow_noprio_r
#define adi_appl_slow_noprio_rw
#define adi_fast_prio0_code l1_text
#define adi_fast_prio2_code l1_text
#define adi_slow_noprio_code
#define adi_fast_prio2_rw   l1_data
#endif /* _LANGUAGE_ASM */
#else /* _SHARED_LIB */

/* For shared libraries move everything to default */
#ifdef _LANGUAGE_ASM
#define adi_fast_prio0_r        .data
#define adi_fastb0_prio0_r      .data
#define adi_fastb1_prio0_r      .data
#define adi_slow_noprio_r       .data
#define adi_fast_prio0_code     .text
#define adi_slow_noprio_code    .text
#else /* _LANGUAGE_ASM */
#define adi_fast_prio0_r
#define adi_fast_prio0_rw
#define adi_fast_prio0_temp
#define adi_fast_prio1_r
#define adi_fastb0_prio0_r
#define adi_fastb0_prio0_rw
#define adi_fastb0_prio0_temp
#define adi_fastb1_prio0_r
#define adi_fastb1_prio0_rw
#define adi_fastb1_prio1_r
#define adi_fastb1_prio0_temp
#define adi_slow_noprio_r
#define adi_appl_slow_noprio_rw
#define adi_fast_prio0_code
#define adi_fast_prio2_code
#define adi_slow_noprio_code
#define adi_fast_prio2_rw
#endif /* _LANGUAGE_ASM */
#endif /* _SHARED_LIB */
#endif /* __VISUALDSPVERSION__ */

/*
** Macro to define section for code and data in C files
*/
#ifdef __VISUALDSPVERSION__
#define xstr(s)                         str(s)
#define str(s)                          #s
#define MEMORY_SECTION(section_name)    section(xstr(section_name))
#elif defined __UCLINUX_FDPIC__
#define MEMORY_SECTION(section_name)    __attribute__(section_name)
#elif defined __UCLINUX_FLAT__
#define MEMORY_SECTION(section_name)
#else
#define MEMORY_SECTION(section_name)
#endif /* not __VISUALDSPVERSION__*/

/*
** Macro to define start of an assembly function
** These are different for different GCC toolchains builds
*/
#if defined __VISUALDSPVERSION__
#define FUNCTION_BEGIN(functname, sect, alignment) \
    .section sect;                                 \
    .global functname;                             \
    .align alignment;                              \
functname:
#elif defined __UCLINUX_FDPIC__
#define FUNCTION_BEGIN(functname, sect, alignment) \
    .section sect;                                 \
    .global functname;                             \
    .align alignment;                              \
functname:
#elif defined __UCLINUX_FLAT__
#define FUNCTION_BEGIN(functname, section, alignment) \
    .global functname;                                \
    .align alignment;                                 \
    .hidden functname;                                \
functname:
#endif /* not __UCLINUX_FDPIC__ and not __VISUALDSPVERSION__*/

/*
** Macro to define end of an assembly function
** These are different for GCC, VDSP builds
*/
#ifdef __VISUALDSPVERSION__
#define FUNCTION_END(functname) functname##.end :
#else
#define FUNCTION_END(functname) .size functname, . -functname
#endif /* not __VISUALDSPVERSION__*/

/*
** Macro to call another function from assmebly
** UCLINUX build (non-flat)  calls the functions
** usning function descriptors
*/

/*
Note P0 and P1 are scratch registers. If some tweeking is
done across asm modules using these registers, this won't work
*/
#if defined __VISUALDSPVERSION__ || defined __UCLINUX_FLAT__
#define FUNCTION_CALL(functname) \
    P0.l = functname;            \
    P0.h = functname;            \
    CALL(P0);
#else /* (__VISUALDSPVERSION__) || (__UCLINUX_FLAT__) */

/* Using ARGOFFSET=12 bytes to store registers.
   If ARGOFFSET=0, then problem. But this is not the case in UCLINUX */
#define FUNCTION_CALL(functname)              \
    [SP + 0] = P3;                            \
    P0 = [P3 + functname##@FUNCDESC_GOT17M4]; \
    P3 = [P0 + 4];                            \
    P0 = [P0];                                \
    CALL(P0);                                 \
    P3 = [SP + 0];
#endif /* (__VISUALDSPVERSION__) || (__UCLINUX_FLAT__) */

/*
** Macro to access a global symbol from assembly
** uClinux build (FDPIC) uses descriptors for access
*/
#if defined __VISUALDSPVERSION__ || defined __UCLINUX_FLAT__
#define GET_POINTER(buffer, p_register) \
    p_register##.l = buffer;            \
    p_register##.h = buffer;
#elif defined __UCLINUX_FDPIC__
#define GET_POINTER(buffer, p_register) p_register = [P3 + buffer##@GOT17M4];
#endif /* (__VISUALDSPVERSION__) || (__UCLINUX_FLAT__) */

/*
** Macro for Aligning buffers
** Currently this is same
** for all tool chains
*/
#if defined __UCLINUX_FLAT__ || defined __UCLINUX_FDPIC__
#define ALIGN(x)    __attribute__((aligned(x)))
#else /*__VISUALDSPVERSION__ */
#define ALIGN(x)
#endif /* not  __VISUALDSPVERSION__ */

/*
** GCC tool chain does not support .import feature.
** Offset of a field within a structure and size of structure
** have to be calculated explicitly
*/
#ifdef __VISUALDSPVERSION__
#define IMPORT(x)   .import x;
#else /*__VISUALDSPVERSION__ */
#define IMPORT(x)
#define OFFSETOF(x, y)  x##_##y
#define SIZEOF(x)       SIZEOF##x
#endif /* not  __VISUALDSPVERSION__ */

//#if defined _LANGUAGE_C
#ifdef __cplusplus
extern
"C"
{
#endif /* __cplusplus */

/*=============  D A T A    T Y P E S   =============*/

/*
** These are not provided by stdint.h
**
*/
#ifndef __VISUALDSPVERSION__

/* Premitives for non vdsp builds*/
typedef char                char_t;     /* plain 8 bit character     */

typedef unsigned char       uint8_t;    /* unsigned 8 bit integer    */
typedef unsigned short      uint16_t;   /* unsigned 16 bit integer   */
typedef unsigned int        uint32_t;   /* unsigned 32 bit integer   */
typedef unsigned long long  uint64_t;   /* unsigned 64 bit integer   */

typedef signed char         int8_t;     /* signed 8 bit integer      */
typedef signed short        int16_t;    /* signed 16 bit integer     */
typedef signed int          int32_t;    /* signed 32 bit integer     */
typedef signed long long    int64_t;    /* signed 64 bit integer     */

typedef float               float32_t;  /* 32 bit floating point     */
typedef double              float64_t;  /* 64 bit floating point     */
#endif /* not  __VISUALDSPVERSION__ */

#ifdef __cplusplus
}

#endif /* __cplusplus */
//#endif /* not _LANGUAGE_C */

#ifdef MISRA_RULES
#pragma diag(pop)
#endif
#endif /* __ADI_TOOL_CHAIN_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/tags/release_1_4_0_imagetoolbox_111510/include/adi_tool_chain.h $
**
*/
