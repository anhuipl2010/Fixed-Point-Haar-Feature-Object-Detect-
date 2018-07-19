/*****************************************************************************
Copyright (c) 1996 - 2012 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 6854 $
$Date: 2012-01-27 20:27:54 +0530 (Fri, 27 Jan 2012) $

Title       : adi_float16.h

Description : C prototype functions for C++ float16 class.
                ** float16 is a 32-bit type. Exponent is low-half.
                ** Mantissa is high half:
                ** s mmm mmmm mmmm mmmm s eee eeee eeee eeee
                ** Exponent is unbiased, and there is no hidden bit,
                ** numbers are normalised to 0.x, not 1.x.

*****************************************************************************/

#if defined(_WIN32) || defined (_WIN64)
#include <windows.h>
#pragma warning(disable:4068)
#pragma warning(disable:4996)
#endif

#pragma once
#ifndef __NO_BUILTIN
#pragma system_header /* float16.h */
#endif

#ifndef __ADI_FLOAT16_H__
#define __ADI_FLOAT16_H__

#define float32_t float
#include "adi_tool_chain.h"

#ifdef _MISRA_RULES
#pragma diag(push)
#pragma diag(suppress:misra_rule_5_2:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_5_3:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_5_4:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_5_5:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_5_6:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_5_7:"ADI header will re-use identifiers")
#pragma diag(suppress:misra_rule_6_3:"ADI header allows use of basic types")
#pragma diag(suppress:misra_rule_8_1:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_8_5:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_8_10:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_10_1_a:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_10_1_b:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_10_1_d:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_12_1:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_12_6:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_12_7:"ADI header - use of inline function")
#pragma diag(suppress:misra_rule_12_12:"ADI header uses float in union")
#pragma diag(suppress:misra_rule_16_3:"ADI header, identifiers not needed on all parameters")
#pragma diag(suppress:misra_rule_18_4:"ADI header allows unions")
#pragma diag(suppress:misra_rule_19_1:"ADI header allows includes after statements")
#pragma diag(suppress:misra_rule_19_7:"ADI header - use of macros for asm codes")
#endif /* _MISRA_RULES */

#if defined __VISUALDSPVERSION__ || defined __CCESVERSION__
#define shift_left(nNum,nShift)\
    ((nNum) << (nShift))
#define shift_right(nNum,nShift)\
    ((nNum) >> (nShift))
#else /* host */
#pragma inline
static int32_t shift_left(int32_t nNum, uint32_t nShift)
{
    int32_t nResult;
    if(nShift >= 32)
    {
        if(nNum >= 0)
        {
            nResult = 0;
        }
        else
        {
            nResult = -1;
        }
    }
    else
    {
        nResult = nNum << nShift;
    }
    return nResult;
}

#pragma pure
#pragma inline
static int32_t shift_right(int32_t nNum, uint32_t nShift)
{
    int32_t nResult;
    if(nShift >= 32)
    {
        if(nNum >= 0)
        {
            nResult = 0;
        }
        else
        {
            nResult = -1;
        }
    }
    else
    {
        nResult = nNum >> nShift;
    }
    return nResult;
}
#endif

#if defined __VISUALDSPVERSION__ || defined __CCESVERSION__
#include <fract.h>
#include <fract2float_conv.h>
#include <builtins.h>
#else
#if 0
   int16_t exponent = (int16_t)norm_fr1x16(fl.s.m);
   fl.s.m = mult_fr1x16(x.s.m, y.s.m);
   ie = max(xe, ye);
   n = norm_fr1x32(im);
   fl.s.e = sub_fr1x16(ie, (n - 16));  /* Exponent is 16-bit saturated */
   fl.s.m = negate_fr1x16(fl.s.m);
   fl.s.m = abs_fr1x16(fl.s.m);
#endif

#pragma inline
static int16_t norm_fr1x16(int16_t x)
{
    int16_t     n = 0;
    int32_t     i, bit;
    uint16_t    _x = (uint16_t)x;
    bit = _x >> 15;
    for (i=1; i<15; i++)
    {
        if ((((uint16_t)(_x << i)) >> 15) == bit)
        {
                n++;
        }
        else
        {
            break;
        }
    }
    return n;
}

#pragma inline
static int16_t norm_fr1x32(int32_t x)
{
    int16_t n = 0;
    uint32_t i,y;
    uint32_t stat = (x & 0x80000000L)>>31;

    for (i = 1; i<=31;i++)
    {
        y = (x & (1<<(31 - i)) ) >> (31- i);
        if (y != stat)
        {
            break;
        }
        n++;
    }
    return n;
}

#pragma inline
static int16_t sub_fr1x16(int16_t x, int16_t y)
{
    int16_t n;

    n = x - y;
    if (x > 0 && y < 0)
    {
        if (n < 0)
        {
            n = 0x7FFF;
        }
    }

    if (x < 0 && y > 0)
    {
        if (n > 0)
        {
            n = 0x8000;
        }
    }
    return n;
}

#pragma inline
static int16_t negate_fr1x16(int16_t x)
{
    int32_t y = 0,z;

    z = y - x;
    if(x > 0 && z >0)
    {
        z = 0x8000;
    }
    else if(x < 0 && z < 0)
    {
        z = 0x7FFF;
    }
    else
    {
        /* Blank else */
    }
    return (int16_t)z;
}

#pragma inline
static uint16_t abs_fr1x16(int16_t x)
{
    uint16_t z;

    if(x >= 0)
    {
        z = x;
    }
    else
    {
        z = (uint16_t)(-x);;
    }

    return z;
}

#pragma inline
static int16_t mult_fr1x16(int16_t x, int16_t y)
{
    int32_t     _x = x, _y = y;
    uint32_t    _xy;
    int16_t     m;

    _xy = (uint32_t)(_x * _y);
    switch (_xy >> 30)
    {
        case 0:
        case 3:
            m = (int16_t)((_xy << 1) >> 16);
            break;
        case 1:
            m = 0x7fff;
            break;
        case 2:
            m = 0x8000;
            break;
    }
    return m;
}

#endif

#define PlusInf_fl16  0x7fffffff
#define NegInf_fl16   0x8000ffff
#define NaN_fl16      0x00008000

#ifdef _FLOAT16_NO_INLINE

typedef int32_t float16;

#ifdef __cplusplus
extern "C" {
#endif

#if defined __VISUALDSPVERSION__  || defined __CCESVERSION__
float16 fr16_to_fl16(fract16);
fract16 fl16_to_fr16(float16);
#endif
float16 norm_fl16(float16 fl);
float16 add_fl16(float16, float16);
float16 sub_fl16(float16, float16);
float16 mul_fl16(float16, float16);
float16 div_fl16(float16, float16);
float16 negate_fl16(float16);
float16 abs_fl16(float16);
int32_t cmp_fl16(float16, float16);
float16 fl_to_fl16(float32_t);
int32_t fits_in_fl16(float32_t);
float32_t fl16_to_fl(float16);

#ifdef __cplusplus
}
#endif

#else

typedef union
{
    int32_t l;          /* for simple initialisations  */
    uint32_t u;         /* for conversions             */
    struct
    {
        int16_t e;      /* low half                    */
        int16_t m;      /* high half                   */
    } s;
    float32_t f;        /* for conversions             */
} float16;

#if defined __VISUALDSPVERSION__  || defined __CCESVERSION__
static float16 fr16_to_fl16(fract16);
static fract16 fl16_to_fr16(float16);
#endif
static float16 norm_fl16(float16 fl);
static float16 add_fl16(float16, float16);
static float16 sub_fl16(float16, float16);
static float16 mul_fl16(float16, float16);
static float16 div_fl16(float16, float16);
static float16 negate_fl16(float16);
static float16 abs_fl16(float16);
static int32_t cmp_fl16(float16, float16);
static float16 fl_to_fl16(float32_t);
static int32_t fits_in_fl16(float32_t);
static float32_t fl16_to_fl(float16);

#if defined __VISUALDSPVERSION__  || defined __CCESVERSION__
#pragma inline
#pragma always_inline
static float16 fr16_to_fl16(fract16 fr)
{
    return fl_to_fl16(fr16_to_float(fr));
}

#pragma inline
#pragma always_inline
static fract16 fl16_to_fr16(float16 fl)
{
    fract16 fraction = fl.s.m;
    int16_t exponent = fl.s.e;

    return shl_fr1x16(fraction, exponent);
}
#endif

#pragma inline
#pragma always_inline
static float16 norm_fl16(float16 fl)
{
    if (fl.s.m != 0)
    {
        int16_t exponent = (int16_t)norm_fr1x16(fl.s.m);
        /* fl.s.m <<= exponent; */
        fl.s.m = shift_left((int32_t)fl.s.m, exponent);
        fl.s.e -= exponent;
    }
    else
    {
        fl.s.e = 0;
    }
    return fl;
}

#pragma inline
#pragma always_inline
static float16 add_fl16(float16 x, float16 y)
{
    float16 fl;     /* Result */
    int16_t ie;     /* Intermediate exponent */
    int32_t im;     /* Intermediate mantissa */
    int16_t n;      /* Normalization distance */

    /* Extract mantissa and exponents from the inputs.  This is an optimization. */
    int32_t xe = x.s.e, ye = y.s.e;
    int32_t xm = x.s.m, ym = y.s.m;

    /* Zero is a tiny number, so make it have a tiny exponent. */
    if (xm == 0)
    {
        xe = -32768;
    }

    if (ym == 0)
    {
        ye = -32768;
    }

    /* Do the add.  Note that the shift might be larger than 31 bits,
     but the Blackfin handles it correctly. */
    ie = max(xe, ye);
    im = (int32_t)shift_right(xm,(ie - xe)) + shift_right(ym,(ie - ye));

    /* Normalize the result in 32-bit int. */
    n = norm_fr1x32(im);
    fl.l = shift_left(im , n);          /* Keep only the upper 16 bits */
    fl.s.e = sub_fr1x16(ie, (n - 16));  /* Exponent is 16-bit saturated */

    /* Convert "zero" to standard form. */
    if (fl.s.m == 0)
    {
        fl.s.e = 0;
    }

    return fl;
}

#pragma inline
#pragma always_inline
static float16 sub_fl16(float16 x, float16 y)
{
    return add_fl16(x, negate_fl16(y));
}

#pragma inline
#pragma always_inline
static float16 mul_fl16(float16 x, float16 y)
{
    float16 fl;

    fl.s.e  = x.s.e + y.s.e;
    fl.s.m = mult_fr1x16(x.s.m, y.s.m);
    if (fl.s.m != 0)
    {
        int16_t exponent = (int16_t)norm_fr1x16(fl.s.m);
        /*fl.s.m <<= exponent;*/
        fl.s.m = shift_left((int32_t)fl.s.m, exponent);
        fl.s.e -= exponent;
    }
    else
    {
        fl.s.e = 0;
    }
    return fl;
}

#pragma inline
#pragma always_inline
static float16 div_fl16(float16 x, float16 y)
{
    int32_t         i;
    int32_t         niters = 15;
    const int32_t   one = 0x40000001;
    float16         fl;
    int32_t         xneg = x.l < 0;
    int32_t         yneg = y.l < 0;
    uint16_t        r = 0u;
    uint16_t        d = (uint16_t)x.s.m;
    uint16_t        q = (uint16_t)y.s.m;
    int16_t         tmp;

    if (y.l == 0)
    {
        fl.l=NaN_fl16;
    }
    else if (y.l == one)
    {
        fl = x;
    }
    else if (x.l == y.l)
    {
        fl.l = one;
    }
    else
    {
        fl.s.e = x.s.e - y.s.e;
        if (xneg != 0)
        {
            tmp = -(int16_t)d;
            d = (uint16_t)tmp;
        }
        if (yneg != 0)
        {
            tmp = -(int16_t)q;
            q = (uint16_t)tmp;
        }
        for (i = 0; i < niters; i++)
        {
            if (q <= d)
            {
                d -= q;
                r |= 1u;
            }
            d <<= 1;
            r <<= 1;
        }
        d = r;
        if ((d & 0x8000u) != 0)
        {
            /* overflow */
            d >>= 1;
            fl.s.e++;
        }
        if (xneg != yneg)
        {
            tmp = -(int16_t)d;
            d = (uint16_t)tmp;
        }
        fl.s.m = (int16_t)d;       /* truncation  */
        if (fl.s.m != 0)
        {
            int16_t exponent = (int16_t)norm_fr1x16(fl.s.m);
            /*fl.s.m <<= exponent;*/
            fl.s.m = shift_left((int32_t)fl.s.m, exponent);
            fl.s.e -= exponent;
        }
        else
        {
            fl.s.e = 0;
        }
    }
    return fl;
}

#pragma inline
#pragma always_inline
static float16 negate_fl16(float16 fl)
{
    fl.s.m = negate_fr1x16(fl.s.m);
    return fl;
}

#pragma inline
#pragma always_inline
static float16 abs_fl16(float16 fl)
{
    fl.s.m = abs_fr1x16(fl.s.m);
    return fl;
}

#pragma inline
#pragma always_inline
static int32_t cmp_fl16(float16 x, float16 y)
{
    /* x < y  => negative
       x == y =>  0
       x > y  => positive */

  int16_t ie;   /* Intermediate exponent */

    /* Extract mantissa and exponents from the inputs.  This is an optimization. */
    int32_t xe = x.s.e, ye = y.s.e;
    int32_t xm = x.s.m, ym = y.s.m;

    /* Zero is a tiny number, so make it have a tiny exponent. */
    if (xm == 0)
    {
        xe = -32768;
    }
    if (ym == 0)
    {
        ye = -32768;
    }

    /* Do the compare.  Note that the shift might be larger than 31 bits,
     but the Blackfin handles it correctly. */
    ie = max(xe, ye);
    return (int32_t)shift_right(xm,(ie - xe)) - shift_right(ym,(ie - ye));
}

#pragma inline
#pragma always_inline
static float16 fl_to_fl16(float32_t f)
{
    float16  fl;
    uint32_t sign, uexp;
    int32_t  exponent;

    fl.f = f;
    if ((fl.u << 1) == 0)
    {
        /* -0.0 or +0.0 - needs special handling */
        fl.u = 0ul;
    }
    else
    {
        sign = (fl.u >> 31) << 31;
        uexp = (fl.u << 1) >> 24;
        uexp = uexp - 127u;
        exponent = (int32_t)uexp;
        exponent += 1;   /* because we have to include the hidden bit */
        fl. u |= (uint32_t)1<<23;   /* add the hidden bit */
        fl.u <<= 8;   /* move 24-bit mantissa to top, to remove exp & sign */
        fl.u >>= 1;   /* back one space, for sign */
#ifdef DEBUG
        if (fl.s.e)
        {
            printf("Warning: precision lost %08x\n", fl.s.e);
        }
#endif
        fl.s.e = (int16_t)exponent;
        if (sign != 0)
        {
            fl.s.m = -fl.s.m;
        }
    }
    return fl;
}

#pragma inline
#pragma always_inline
static int32_t fits_in_fl16(float32_t f)
{
    float16 fl;

    fl.f = f;
    return (fl.u & 0xffu) == 0;
}

#pragma inline
#pragma always_inline
static float32_t fl16_to_fl(float16 fl)
{
    uint32_t    exponent, sign;

    if (fl.u != 0)
    {
        int16_t temp;
        sign = (fl.u >> 31) << 31;
        temp = fl.s.e + 127 - 1;   /* remove one, for hidden bit */
        exponent = (uint32_t)temp;
        if (((uint16_t)fl.s.m & 0xffffu) == 0x8000u)
        {
            fl.s.m = 0x4000;
            exponent++;
        }
        else if (sign != 0)
        {
            fl.s.m = -fl.s.m;
        }
        else
        {
            ; /* required by MISRA */
        }
        fl.s.e = 0;
        fl.u <<= 2;   /* remove sign bit and hidden bit*/
        fl.u >>= 9;
        exponent <<= 23;
        fl.u = sign | exponent | fl.u;
    }
    return fl.f;
}

#endif /*  _FLOAT16_NO_INLINE */

#ifdef _MISRA_RULES
#pragma diag(pop)
#endif /* _MISRA_RULES */

#endif /* __ADI_FLOAT16_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/trunk/image_processing_toolbox/include/adi_tool_chain.h $
**
*/

