/*****************************************************************************
Copyright (c) 2009 - 2010 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 2599 $
$Date: 2010-05-13 19:15:47 +0530 (Thu, 13 May 2010) $

Title       : adi_contour_mem_alloc.h

Description : MACROs and function prototype declarations of public APIs for
              dynamic memory alloc required by contours

*****************************************************************************/
#ifndef __ADI_CONTOUR_MEM_ALLOC_H__
#define __ADI_CONTOUR_MEM_ALLOC_H__

/*=============  I N C L U D E S   =============*/
#include "adi_image_tool_box_defines.h"

/*==============  D E F I N E S  ===============*/

/*=============  D A T A    T Y P E S   =============*/
#if defined(_LANGUAGE_C)

/* Structure to hold mem alloc function pointers and the private data required by them */
typedef struct _DynamicMemAlloc ADI_CONTOUR_DYNAMIC_MEM_ALLOC;
struct _DynamicMemAlloc
{
    void    *pDynamicMemAllocPrivateData;                       /* Pointer to private data required by memory allocation functions */
    void*   (*pMemAlloc)(ADI_CONTOUR_DYNAMIC_MEM_ALLOC  *pDMA,  /* Pointer to Mem block allocation function */
                         ADI_CONTOUR_MEM_BLK_TYPE       nType,
                         uint32_t                       nCount);
    void    (*pMemFree)(ADI_CONTOUR_DYNAMIC_MEM_ALLOC   *pDMA,  /* Pointer to Mem block free function */
                        void                            *pMem,
                        ADI_CONTOUR_MEM_BLK_TYPE        nType,
                        uint32_t                        nCount);
    void    (*pMemReset)(ADI_CONTOUR_DYNAMIC_MEM_ALLOC  *pDMA,  /* Pointer to function that frees parent mem block
                                                                   along with its sub-mem blocks */
                         void                           *pMem,
                         ADI_CONTOUR_MEM_BLK_TYPE       nType,
                         uint32_t                       nCount);
};

/*=============  E X T E R N A L S  ============*/

/*
**  External Data section
*/

/*
**  External Function Prototypes
*/
extern
ADI_ITB_MODULE_STATUS   adi_DynamicMemInit(
                                           ADI_CONTOUR_DYNAMIC_MEM_ALLOC   *pDMA,
                                           int8_t                          *pMem,
                                           uint32_t                        nSizeOfMem,
                                           uint32_t                        nHeight
                                           );

#else /*  assembly language specific macros and extern declarations. */
#endif /* if !defined(_LANGUAGE_C) */
#endif /* __ADI_CONTOUR_MEM_ALLOC_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/tags/release_1_4_0_imagetoolbox_111510/include/adi_contour_mem_alloc.h $
**
*/
