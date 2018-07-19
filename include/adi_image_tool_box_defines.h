/*****************************************************************************
Copyright (c) 2009 - 2010 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 4213 $
$Date: 2010-11-15 17:18:45 +0530 (Mon, 15 Nov 2010) $

Title       : adi_image_tool_box_defines.h

Description : MACROs and Object data type declarations

*****************************************************************************/
#ifndef __ADI_IMAGE_TOOL_BOX_DEFINES_H__
#define __ADI_IMAGE_TOOL_BOX_DEFINES_H__

/*=============  I N C L U D E S   =============*/
#include "adi_tool_chain.h"
//#if defined _LANGUAGE_C
/* #include <float16.h> */
#include "adi_float16.h"
//#endif /*_LANGUAGE_C  */

/*==============  D E F I N E S  ===============*/
#ifdef _MISRA_RULES
#pragma diag(push)

/*
MISRA Rule 5.6(Adv): No identifier in one name space should have the same
spelling as an identifier in another name space, with the exception
of structure member and union member names.
*/
#pragma diag(suppress : misra_rule_5_6)

/*
MISRA Rule 5.7(Adv): No identifier name should be reused.
*/
#pragma diag(suppress : misra_rule_5_7)
#endif

/* Module codes */
#define E_ADI_MODULE_COLORCONV          0x0000
#define E_ADI_MODULE_CONVOLVE           0x0001
#define E_ADI_MODULE_CORRELATION        0x0002
#define E_ADI_MODULE_MORPHOLOGY         0x0003
#define E_ADI_MODULE_BOUNDARY           0x0004
#define E_ADI_MODULE_SOBEL              0x0005
#define E_ADI_MODULE_STATISTICAL_TOOL   0x0006
#define E_ADI_MODULE_HOUGH_TRANSFORM    0x0007
#define E_ADI_MODULE_INTEGRAL_IMAGE     0x0008
#define E_ADI_MODULE_MATRIXOPERATIONS   0x0009
#define E_ADI_MODULE_IMAGE_FILTERS      0x000A
#define E_ADI_MODULE_VECTOROPERATIONS   0x000B
#define E_ADI_MODULE_PYRAMID            0x000C
#define E_ADI_MODULE_KALMAN_FILTER      0x000D
#define E_ADI_MODULE_OPTICALFLOW        0x000E
#define E_ADI_MODULE_PYR_OPTFL          0x000F
#define E_ADI_MODULE_CONTOURS           0x0010
#define E_ADI_MODULE_HAARFEATURES       0x0011
#define E_ADI_MODULE_RLE                0x0012
#define E_ADI_MODULE_PACKED_RLE         0x0013
#define E_ADI_MODULE_UKF                0x0014
#define E_ADI_MODULE_BAYER              0x0015
#define E_ADI_MODULE_CORNER             0x0016

/* Global(for all modules) status codes */
#define E_ADI_SUCCESS           0
#define E_ADI_FAILURE           1
#define E_ADI_INSUFFICIENT_MEM  2

/* status code specific to Matrix Operations */
#define E_ADI_MAT_DIM_DIF       3
#define E_ADI_MAT_OUT_BOUND     4
#define E_ADI_MAT_SINGULAR      5
#define E_ADI_MAT_NO_CONVERGE   6
#define E_ADI_MAT_DIM_ONE       7
#define E_ADI_MAT_DIM_TWO       8

/* status code specific to Kalman Filter */
#define E_ADI_KALMAN_A_DIM_DIF  3
#define E_ADI_KALMAN_H_DIM_DIF  4
#define E_ADI_KALMAN_P_DIM_DIF  5
#define E_ADI_KALMAN_B_DIM_DIF  6
#define E_ADI_KALMAN_Q_DIM_DIF  7
#define E_ADI_KALMAN_R_DIM_DIF  8
#define E_ADI_KALMAN_K_DIM_DIF  9
#define E_ADI_KALMAN_W_DIM_DIF  10
#define E_ADI_KALMAN_V_DIM_DIF  11
#define E_ADI_KALMAN_JACO_ERR   20
#define E_ADI_KALMAN_PRED_ERR   21

/* status code specific to Pyramidal optical flow */
#define E_ADI_PYR_OPTFL_LOST    3   /* feature lost */
#define E_ADI_PYR_OPTFL_OOB     4   /* feature out of bound */
#define E_ADI_PYR_OPTFL_MAXITR  5   /* max iteration */

/* status code specific to Contours */
#define E_ADI_CONTOURS_NOTDONE  3

/* status code specific to packed RLE/RLD */
#define E_ADI_PKD_RLE_MODE_UNKNOWN  3
#define E_ADI_PKD_RLE_MAXCNT_REACH  4

/* status code specific to UKF */
#define E_ADI_UKF_CHOL_FAIL 3

/* Erosion/Dilation Mask Type codes */
#define E_ADI_MASK_CROSS        0
#define E_ADI_MASK_RECTANGLE    1
#define E_ADI_MASK_CUSTOM       2

/* Vectoroperations Norm Type codes */
#define E_ADI_VECTOR_NORM_L1    0
#define E_ADI_VECTOR_NORM_L2    1

/* Distance Tranform Metric Type codes */
#define E_ADI_DT_CHESSBOARD 0
#define E_ADI_DT_CITYBLOCK  1

/* Distance Tranform Process Type codes */
#define E_ADI_DT_FORWARD    0
#define E_ADI_DT_BACKWARD   1

/* Covariance Option Codes */
#define E_ADI_COVAR_NO_AVG_NO_SCALE 0x0000
#define E_ADI_COVAR_AVG_NO_SCALE    0x0001
#define E_ADI_COVAR_NO_AVG_SCALE    0x0002
#define E_ADI_COVAR_AVG_SCALE       0x0003

/* Contour memory block type codes */
#define E_ADI_CONTOUR_RUN_LEN_HDR       1   /* to choose memory block ADI_CONT_RUN_LEN_HDR */
#define E_ADI_CONTOUR_BOUNDING_RECT     2   /* to choose memory block ADI_CONT_BOUNDING_RECTANGLE */
#define E_ADI_CONTOUR_RUN_LEN_NODE      3   /* to choose memory block ADI_CONT_RUN_LEN_NODE */
#define E_ADI_CONTOUR_SEG_HDR           4   /* to choose memory block ADI_CONT_SEGMENT_HDR */
#define E_ADI_CONTOUR_EQU_SEG_HDR       5   /* to choose memory block ADI_CONT_EQU_SEGMENT_HDR */
#define E_ADI_CONTOUR_EQU_SEG_NODE      6   /* to choose memory block ADI_CONT_EQU_SEGMENT_NODE */
#define E_ADI_CONTOUR_SEG_HDR_ROW_WISE  7   /* to choose memory block ADI_CONT_SEGMENT_HDR_ROW_WISE */
#define E_ADI_CONTOUR_HDR               8   /* to choose memory block ADI_CONT_HDR */
#define E_ADI_CONTOUR_NODE              9   /* to choose memory block ADI_CONT_NODE */

/* Packet type codes */
#define E_ADI_MODE_32_16W_16R   0
#define E_ADI_MODE_32_3I_13R    1
#define E_ADI_MODE_32_4I_12R    2
#define E_ADI_MODE_32_3I_5R     3
#define E_ADI_MODE_32_4I_4R     4
#define E_ADI_MODE_32_4I_21R    5

/* Bayer demosaicing algo codes */
#define E_ADI_BAYER2RGB_BOTTOM_RIGHT    0   /* Nearest Neighbor (Bottom Right) */
#define E_ADI_BAYER2RGB_TOP_LEFT        1   /* Nearest Neighbor (Top Left) */
#define E_ADI_BAYER2RGB_BILINEAR        2   /* Bilinear */

/*=============  D A T A    T Y P E S   =============*/
//#if defined(_LANGUAGE_C)

/* define additional data types */
typedef double  double_t;

/* Module code Enumerations */
typedef enum
{
    ADI_ITB_MODULE_COLORCONV        = E_ADI_MODULE_COLORCONV,
    ADI_ITB_MODULE_CONVOLVE         = E_ADI_MODULE_CONVOLVE,
    ADI_ITB_MODULE_CORRELATION      = E_ADI_MODULE_CORRELATION,
    ADI_ITB_MODULE_MORPHOLOGY       = E_ADI_MODULE_MORPHOLOGY,
    ADI_ITB_MODULE_BOUNDARY         = E_ADI_MODULE_BOUNDARY,
    ADI_ITB_MODULE_SOBEL            = E_ADI_MODULE_SOBEL,
    ADI_ITB_MODULE_STATISTICAL_TOOL = E_ADI_MODULE_STATISTICAL_TOOL,
    ADI_ITB_MODULE_HOUGH_TRANSFORM  = E_ADI_MODULE_HOUGH_TRANSFORM,
    ADI_ITB_MODULE_INTEGRAL_IMAGE   = E_ADI_MODULE_INTEGRAL_IMAGE,
    ADI_ITB_MODULE_MATRIXOPERATIONS = E_ADI_MODULE_MATRIXOPERATIONS,
    ADI_ITB_MODULE_IMAGE_FILTERS    = E_ADI_MODULE_IMAGE_FILTERS,
    ADI_ITB_MODULE_VECTOROPERATIONS = E_ADI_MODULE_VECTOROPERATIONS,
    ADI_ITB_MODULE_PYRAMID          = E_ADI_MODULE_PYRAMID,
    ADI_ITB_MODULE_KALMAN_FILTER    = E_ADI_MODULE_KALMAN_FILTER,
    ADI_ITB_MODULE_OPTICALFLOW      = E_ADI_MODULE_OPTICALFLOW,
    ADI_ITB_MODULE_PYR_OPTFL        = E_ADI_MODULE_PYR_OPTFL,
    ADI_ITB_MODULE_CONTOURS         = E_ADI_MODULE_CONTOURS,
    ADI_ITB_MODULE_HAARFEATURES     = E_ADI_MODULE_HAARFEATURES,
    ADI_ITB_MODULE_RLE              = E_ADI_MODULE_RLE,
    ADI_ITB_MODULE_PACKED_RLE       = E_ADI_MODULE_PACKED_RLE,
    ADI_ITB_MODULE_UKF              = E_ADI_MODULE_UKF,
    ADI_ITB_MODULE_BAYER            = E_ADI_MODULE_BAYER,
    ADI_ITB_MODULE_CORNER           = E_ADI_MODULE_CORNER
} ADI_ITB_MODULE;

/* Status code Enumerations */
typedef enum
{
    ADI_ITB_STATUS_SUCCESS          = E_ADI_SUCCESS,
    ADI_ITB_STATUS_FAILURE          = E_ADI_FAILURE,
    ADI_ITB_STATUS_INSUFFICIENT_MEM = E_ADI_INSUFFICIENT_MEM,
    ADI_ITB_STATUS_MAT_DIM_DIF      = E_ADI_MAT_DIM_DIF,
    ADI_ITB_STATUS_MAT_OUT_BOUND    = E_ADI_MAT_OUT_BOUND,
    ADI_ITB_STATUS_MAT_SINGULAR     = E_ADI_MAT_SINGULAR,
    ADI_ITB_STATUS_MAT_NO_CONVERGE  = E_ADI_MAT_NO_CONVERGE,
    ADI_ITB_STATUS_MAT_DIM_ONE      = E_ADI_MAT_DIM_ONE,
    ADI_ITB_STATUS_MAT_DIM_TWO      = E_ADI_MAT_DIM_TWO,
    ADI_ITB_STATUS_KALMAN_A_DIM_DIF = E_ADI_KALMAN_A_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_H_DIM_DIF = E_ADI_KALMAN_H_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_P_DIM_DIF = E_ADI_KALMAN_P_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_B_DIM_DIF = E_ADI_KALMAN_B_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_Q_DIM_DIF = E_ADI_KALMAN_Q_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_R_DIM_DIF = E_ADI_KALMAN_R_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_K_DIM_DIF = E_ADI_KALMAN_K_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_W_DIM_DIF = E_ADI_KALMAN_W_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_V_DIM_DIF = E_ADI_KALMAN_V_DIM_DIF,
    ADI_ITB_STATUS_KALMAN_JACO_ERR  = E_ADI_KALMAN_JACO_ERR,
    ADI_ITB_STATUS_KALMAN_PRED_ERR  = E_ADI_KALMAN_PRED_ERR,
    ADI_ITB_STATUS_PYR_OPTFL_LOST   = E_ADI_PYR_OPTFL_LOST,
    ADI_ITB_STATUS_PYR_OPTFL_OOB    = E_ADI_PYR_OPTFL_OOB,
    ADI_ITB_STATUS_PYR_OPTFL_MAXITR = E_ADI_PYR_OPTFL_MAXITR,
    ADI_ITB_STATUS_CONTOURS_NOTDONE = E_ADI_CONTOURS_NOTDONE,
    ADI_ITB_STATUS_RLE_MODE_UNKNOWN = E_ADI_PKD_RLE_MODE_UNKNOWN,
    ADI_ITB_STATUS_RLE_MAXCNT_REACH = E_ADI_PKD_RLE_MAXCNT_REACH,
    ADI_ITB_STATUS_UKF_CHOL_FAIL    = E_ADI_UKF_CHOL_FAIL
} ADI_ITB_STATUS;

/* Status of Module (= ADI_ITB_MODULE << 16 | ADI_ITB_STATUS) */
typedef int32_t ADI_ITB_MODULE_STATUS;

/* Erosion/Dilation Mask Type Enumerations */
typedef enum
{
    ADI_MASK_CROSS                  = E_ADI_MASK_CROSS,
    ADI_MASK_RECTANGLE              = E_ADI_MASK_RECTANGLE,
    ADI_MASK_CUSTOM                 = E_ADI_MASK_CUSTOM
} ADI_MASK_OPTION;

/* Vectoroperations Norm Type codes */
typedef enum
{
    ADI_VECTOR_L1_NORM              = E_ADI_VECTOR_NORM_L1,
    ADI_VECTOR_L2_NORM              = E_ADI_VECTOR_NORM_L2
} ADI_VECTOR_NORM_TYPE;

/* Distance Transform Metric Type Enumerations */
typedef enum
{
    ADI_DT_CHESSBOARD               = E_ADI_DT_CHESSBOARD,
    ADI_DT_CITYBLOCK                = E_ADI_DT_CITYBLOCK
} ADI_DT_METRIC;

/* Distance Transform Process Type Enumerations */
typedef enum
{
    ADI_DT_FORWARD                  = E_ADI_DT_FORWARD,
    ADI_DT_BACKWARD                 = E_ADI_DT_BACKWARD
} ADI_DT_PROCESS;

/* Covariance Option Enumerations */
typedef enum
{
    ADI_COVAR_NO_AVG_NO_SCALE       = E_ADI_COVAR_NO_AVG_NO_SCALE,
    ADI_COVAR_AVG_NO_SCALE          = E_ADI_COVAR_AVG_NO_SCALE,
    ADI_COVAR_NO_AVG_SCALE          = E_ADI_COVAR_NO_AVG_SCALE,
    ADI_COVAR_AVG_SCALE             = E_ADI_COVAR_AVG_SCALE
} ADI_COVAR_OPTION;

/* Contour memory block type Enumerations */
typedef enum
{
    ADI_CONTOUR_RUN_LEN_HDR         = E_ADI_CONTOUR_RUN_LEN_HDR,
    ADI_CONTOUR_BOUNDING_RECT       = E_ADI_CONTOUR_BOUNDING_RECT,
    ADI_CONTOUR_RUN_LEN_NODE        = E_ADI_CONTOUR_RUN_LEN_NODE,
    ADI_CONTOUR_SEG_HDR             = E_ADI_CONTOUR_SEG_HDR,
    ADI_CONTOUR_EQU_SEG_HDR         = E_ADI_CONTOUR_EQU_SEG_HDR,
    ADI_CONTOUR_EQU_SEG_NODE        = E_ADI_CONTOUR_EQU_SEG_NODE,
    ADI_CONTOUR_SEG_HDR_ROW_WISE    = E_ADI_CONTOUR_SEG_HDR_ROW_WISE,
    ADI_CONTOUR_HDR                 = E_ADI_CONTOUR_HDR,
    ADI_CONTOUR_NODE                = E_ADI_CONTOUR_NODE
} ADI_CONTOUR_MEM_BLK_TYPE;

/* Packet type code Enumerations */
typedef enum
{
    ADI_PACKET_MODE_32_16W_16R      = E_ADI_MODE_32_16W_16R,
    ADI_PACKET_MODE_32_3I_13R       = E_ADI_MODE_32_3I_13R,
    ADI_PACKET_MODE_32_4I_12R       = E_ADI_MODE_32_4I_12R,
    ADI_PACKET_MODE_32_3I_5R        = E_ADI_MODE_32_3I_5R,
    ADI_PACKET_MODE_32_4I_4R        = E_ADI_MODE_32_4I_4R,
    ADI_PACKET_MODE_32_4I_21R       = E_ADI_MODE_32_4I_21R
} ADI_PACKET_MODE;

/* Bayer demosaicing algo code Enumerations */
typedef enum
{
    ADI_BAYER2RGB_BOTTOM_RIGHT      = E_ADI_BAYER2RGB_BOTTOM_RIGHT,
    ADI_BAYER2RGB_TOP_LEFT          = E_ADI_BAYER2RGB_TOP_LEFT,
    ADI_BAYER2RGB_BILINEAR          = E_ADI_BAYER2RGB_BILINEAR
} ADI_BAYER_ALGO;

/* ADI_HOUGH_DATA is the handle for Hough data populated during initialization */
typedef struct _adiHoughData
{
    const uint8_t   *pInBuff;           /* Input Buffer pointer.*/
    uint16_t        nImageRoiHeight;    /* Height of Image/ROI */
    uint16_t        nImageRoiWidth;     /* Width of Image/ROI */
    int16_t         nThetaStart;        /* Starting angle of theta. Used for sine table generation. */
    int16_t         nThetaEnd;          /* End value of theta. Used for sine table generation */
    uint16_t        nRhoStep;           /* Rho step in accumulator array. */
    uint16_t        nLineNo;            /* Input line no. User should populate with
                                           actual line no he is sending during voting process. */
    uint16_t        nThetaStep;         /* Step for theta. */
    int16_t         *pSinBuff;          /* Buffer address where sine values are stored. */
    int16_t         *pCosBuff;          /* Buffer address where cosine values are stored. */
    int16_t         nInvRhoStep;        /* Inverse of rho in 1.15 format.Used internally. */
    uint16_t        nNumRho;            /* Number of Rho Values present in accumulator.Used internally. */
    uint16_t        nNumTheta;          /* Number of theta values  used during voting.Used internally. */
    uint16_t        nImageStride;       /* Image stride added after processing a row. Can be used
                                           for ROI implementation. Default value is 0. */
    uint32_t        nReqdAccSize;       /* Required accumulator size for the given configuration. */
} ADI_HOUGH_DATA;

/* ADI_FLOODFILL_DATA structure will hold statistics about the areas filled. */
typedef struct _adiFloodFillData
{
    uint32_t    nArea;      /* The nArea is the area of "filled" of the component.*/
    uint16_t    nXcord;     /* The nXcord is the x coordinate pixel from where new value is filled.*/
    uint16_t    nYcord;     /* The nYcord is the x coordinate pixel from where new value is filled.*/
    uint16_t    nWidth;     /* The nWidth is width of the filled area.*/
    uint16_t    nHeight;    /* The nHeight is height of the filled area.*/
    uint8_t     nValue;     /* The nValue is the new value which is filled in image.*/
} ADI_FLOODFILL_DATA;

/* ADI_MAT will hold Matrix data */
typedef struct _adiMat
{
    int8_t      *pMat;  /* pointer to matrix */
    uint32_t    nRows;  /* number of rows in the matrix */
    uint32_t    nCols;  /* number of columns in the matrix */
} ADI_MAT;

/* ADI_KALMAN will hold state and covariance data for current and predicted states */
typedef struct _adiKalman
{
    ADI_MAT     *pMatA;         /* state transition matrix */
    ADI_MAT     *pMatH;         /* measurement/observation matrix */
    ADI_MAT     *pMatX;         /* Posterior state vector */
    ADI_MAT     *pMatP;         /* Posterior Co-variance matrix */
    ADI_MAT     *pMatB;         /* Control matrix */
    ADI_MAT     *pMatQ;         /* Co-variance of State Process Noise */
    ADI_MAT     *pMatR;         /* Co-variance of Measurement Process Noise */
    ADI_MAT     *pMatW;         /* Jacobian of State Transition matrix wrt noise */
    ADI_MAT     *pMatV;         /* Jacobian of Measurement matrix wrt noise */

    ADI_MAT     *pMatXPre;      /* Apriori state vector */
    ADI_MAT     *pMatPPre;      /* Apriori Co-variance matrix */
    ADI_MAT     *pMatK;         /* Kalman Gain Matrix */

    int8_t      *pTemp;         /* Temporary Space for intermediate computations */
    uint32_t    uKalmanGain;    /* Kalman Gain scale Factor */
    uint32_t    nShift;         /* Fixed Point Representation */

    int8_t      bIsIdentityH;   /* measurement matrix is identity matrix */
    int8_t      bIsIdentityA;   /* transition matrix is identity matrix */
    int8_t      bIsIdentityB;   /* Control matrix if present is identity matrix */
    int8_t      bIsIdentityW;   /* Jacobian of state transition matrix wrt noise */
                                /* if present is identity matrix */
    int8_t      bIsIdentityV;   /* Jacobian of Measurement matrix wrt noise */
                                /* if present is identity matrix */

    int8_t      bEnableExt;     /* Enables Extented Kalman Filter */

    /* Function Pointer for Jacobian Calculation */
                                /* Jacobian of state transition function wrt state */
                                /* NULL if state transition is linear */
    void        (*pt2JacoAFunc) (int32_t     *pState,        /* State Vector Pointer */
                                 uint32_t    nStateSize,     /* Number of State */
                                 int32_t     *pControl,      /* Control Vector Pointer */
                                 uint32_t    nControlSize,   /* Dimension of Control Vector */
                                 int32_t     *pJacoA,        /* Jacobian at given input para (Output) */
                                 uint32_t    nShift);        /* Number if bits for fractional part */

                                 /* Jacobian of measurement function wrt to state */
                                 /* NULL if measurement is linear */
    void        (*pt2JacoHFunc) (int32_t     *pState,        /* Pointer to State Vector */
                                 uint32_t    nStateSize,     /* Number of State */
                                 uint32_t    nObserSize,     /* Dimension of Observation */
                                 int32_t     *pJacoH,        /* Pointer to Jacobian Output */
                                 uint32_t    nShift);        /* Number if bits for fractional part */

                                    /* Jacobian of state transition function wrt noise */
                                    /* NULL if W is linear */
    void        (*pt2JacoWFunc) (int32_t    *pState,            /* State Vector Pointer */
                                 uint32_t   nStateSize,         /* Number of State */
                                 int32_t    *pControl,          /* Control Vector Pointer */
                                 uint32_t   nControlSize,       /* Dimension of Control Vector */
                                 uint32_t   nProcessNoiseSize,  /* Process Noise Size */
                                 int32_t    *pJacoW,            /* Jacobian at given input para (Output) */
                                 uint32_t    nShift);            /* Number if bits for fractional part */

                                    /* Jacobian of measurement function wrt to noise */
                                    /* NULL if V is linear */
    void        (*pt2JacoVFunc) (int32_t    *pState,                /* Pointer to State Vector */
                                 uint32_t   nStateSize,             /* Number of State */
                                 uint32_t   nObserSize,             /* Number of Observation */
                                 uint32_t   nMeasurementNoiseSize,  /* Measurement Noise Size */
                                 int32_t    *pJacoV,                /* Pointer to Jacobian Output */
                                 uint32_t    nShift);               /* Number if bits for fractional part */

    /* Function for prediction */
    void        (*pt2PredState) (int32_t     *pState,        /* Pointer to State Vector */
                                 uint32_t    nStateSize,     /* Dimension of State */
                                 int32_t     *pControl,      /* Control Vector Pointer */
                                 uint32_t    nControlSize,   /* Dimension of Control Vector */
                                 int32_t     *pPredState,    /* Predicted State Output */
                                 uint32_t    nShift);        /* Number if bits for fractional part */

    void        (*pt2PredObser) (int32_t     *pState,        /* Pointer to State Vector */
                                 uint32_t    nStateSize,     /* Dimension of State */
                                 uint32_t    nObserSize,     /* Dimension of Observation */
                                 int32_t     *pPredObser,    /* Predicted Observation */
                                 uint32_t    nShift);        /* Number if bits for fractional part */

} ADI_KALMAN;

/* ADI_OPTICALFLOW structure will hold input parameters for block matching */
typedef struct _adiOpticalFlow
{
    uint8_t     *pSrcA8;        /* Pointer to source image-A */
    uint16_t    *pSrcA16;       /* Pointer to source image-A */
    uint8_t     *pSrcB8;        /* Pointer to source image-B */
    uint16_t    *pSrcB16;       /* Pointer to source image-B */
    uint32_t    nImageStep;     /* Image stride OR Image step */
    uint32_t    nImageWidth;    /* Image width */
    uint32_t    nImageHeight;   /* Image height */
    uint32_t    nBlockWidth;    /* Block Width */
    uint32_t    nBlockHeight;   /* Block Height */
    uint32_t    nShiftWidth;    /* Shift Width */
    uint32_t    nShiftHeight;   /* Shift height */
    uint32_t    nMaxRangeWidth; /* Max Range Width */
    uint32_t    nMaxRangeHeight;    /* Max Range Height */
    float16     *pVelocX;           /* Pointer to store Velocity in X direction */
    float16     *pVelocY;           /* Pointer to store velocity in Y direction */
    uint32_t    nVeloStride;        /* stride to store the velocity */
} ADI_OPTICALFLOW, *ADI_OPTICALFLOW_PTR;

/* ADI_OPTICALLKBUFF hold Buffer Information Required for Lucas Kanade */
typedef struct _adiOpticalLKBuff
{
    uint8_t *pBuffer;   /* Pointer to the Buffer */
    uint8_t *pBuffCurr; /* Current Buffer Pointer */
} ADI_OPTICALLKBUFF;

/* ADI_PYR_OPTFL_GD_FEAT structure will hold input parameters for good features */
typedef struct _adiPyrOptflGdFeat
{
    uint16_t    nCoordX;            /* Coordinate X */
    uint16_t    nCoordY;            /* Coordinate Y */

    int32_t     nVal_StatusCode;    /* Score or return code */

    uint16_t    nNewCoordX;         /* New Coordinate X */
    uint16_t    nNewCoordY;         /* New Coordinate Y */
} ADI_PYR_OPTFL_GD_FEAT, *ADI_PYR_OPTFL_GD_FEAT_PTR;

/* ADI_PYR_OPTFL_IMAGE_FEAT structure will hold input parameters for image preprocessing before tracking features */
typedef struct _adiPyrOptflImageFeat
{
    uint8_t     *pImg;          /* Pointer to input image */
    uint32_t    nImageWidth;    /* Image width */
    uint32_t    nImageHeight;   /* Image height */
    uint8_t     *pSmoothImg;    /* Pointer to smoothed output image */
    uint8_t     *pImgPyr;       /* Pointer to image pyramid */
    int16_t     *pGradXImg;     /* Pointer to gradient in x direction for Img */
    int16_t     *pGradYImg;     /* Pointer to gradient in y direction for Img */
    int16_t     *pGradXPyr;     /* Pointer to gradient in x direction for Pyr */
    int16_t     *pGradYPyr;     /* Pointer to gradient in y direction for Pyr */
} ADI_PYR_OPTFL_IMAGE_FEAT, *ADI_PYR_OPTFL_IMAGE_FEAT_PTR;

/* Object type to hold Run-Length encoded node */
typedef struct _RunLenNode ADI_CONT_RUN_LEN_NODE;
struct _RunLenNode
{
    uint16_t                xStart;                 /* Start x co-ordinate for run of 1's */
    uint16_t                xEnd;                   /* End x co-ordinate for run of 1's */
    uint16_t                y;                      /* Y co-ordinate for run of 1's */
    uint16_t                nAddedToContour;        /* place to hold info about node added to contour
                                                       0 : not added
                                                       1 : 1 edge added
                                                       2 : both edge added */
    uint32_t                nSegNum;                /* Segment number of the segment to which this node belongs */
    ADI_CONT_RUN_LEN_NODE   *pNext;                 /* next ADI_CONT_RUN_LEN_NODE node in image */
    ADI_CONT_RUN_LEN_NODE   *pNextInSegmentList;    /* next ADI_CONT_RUN_LEN_NODE node in segment */
};

/* Object type for Run-Length encoded node list header */
typedef struct _RunLenHdr
{
    ADI_CONT_RUN_LEN_NODE   *pNext; /* ADI_CONT_RUN_LEN_NODE list for one row */
} ADI_CONT_RUN_LEN_HDR;

/* Object type to hold Bounding Rectangle of a segment */
typedef struct _BoundingRectangle
{
    uint16_t    nXTopLeft;      /* x co-ordinate of Top-Left point of Bounding Rectangle */
    uint16_t    nYTopLeft;      /* y co-ordinate of Top-Left point of Bounding Rectangle */
    uint16_t    nXBottomRight;  /* x co-ordinate of Bottom-Right point of Bounding Rectangle */
    uint16_t    nYBottomRight;  /* y co-ordinate of Bottom-Right point of Bounding Rectangle */
    uint16_t    nWidth;         /* Width of Bounding Rectangle */
    uint16_t    nHeight;        /* Height of Bounding Rectangle */
} ADI_CONT_BOUNDING_RECTANGLE;

/* Object type to hold Run-Length encoded node list that make a segment */
typedef struct _SegmentHdr  ADI_CONT_SEGMENT_HDR;
struct _SegmentHdr
{
    uint32_t                nSegNum;                /* Self Segment number */
    ADI_CONT_RUN_LEN_NODE   *pNextInSegmentList;    /* next ADI_CONT_RUN_LEN_NODE node in segment */
    ADI_CONT_SEGMENT_HDR    *pDown;                 /* next segment in image */
};

/* Object type to hold Run-Length encoded node list(Row-wise) that make a segment */
typedef struct _SegmentHdrRowWise
{
    ADI_CONT_RUN_LEN_NODE   *pNext; /* next ADI_CONT_RUN_LEN_NODE node in segment in the same row */
} ADI_CONT_SEGMENT_HDR_ROW_WISE;

/* Object type for Equivalent Segment Number */
typedef struct _EquivalenceSegmentNode  ADI_CONT_EQU_SEGMENT_NODE;
struct _EquivalenceSegmentNode
{
    uint32_t                    nSegNum;    /* Equivalent Segment number */
    ADI_CONT_EQU_SEGMENT_NODE   *pPrev;     /* previous Equivalent Segment number node */
    ADI_CONT_EQU_SEGMENT_NODE   *pNext;     /* next Equivalent Segment number node */
};

/* Object type for Equivalent Segment Number list header */
typedef struct _EquivalenceSegmentHdr   ADI_CONT_EQU_SEGMENT_HDR;
struct _EquivalenceSegmentHdr
{
    ADI_CONT_EQU_SEGMENT_NODE   *pNext; /* List of Equivalent Segment Number nodes */
    ADI_CONT_EQU_SEGMENT_HDR    *pDown; /* next Equivalent Segment Number list header */
};

/* Object type for Contour node */
typedef struct _ContourNode ADI_CONT_NODE;
struct _ContourNode
{
    uint16_t        bInUse; /* do not change this place or use it */
    uint16_t        x;      /* x co-ordinate of the point in the contour/freeman code */
    uint16_t        y;      /* y co-ordinate of the point in the contour/num occurrences of freeman code */
    ADI_CONT_NODE   *pNext; /* next/adjacent contour node */
};

/* Object type for Contour header */
typedef struct _ContourHdr  ADI_CONT_HDR;
struct _ContourHdr
{
    uint16_t        bInUse;     /* do not change this place or use it */
    uint16_t        nDirection; /* direction of contour tracing(Top2Bottom or Bottom2Top) */
    ADI_CONT_NODE   *pAnchor;   /* First contour node traced for the segment */
    ADI_CONT_NODE   *pRight;    /* Right most contour node traced for the segment */
    ADI_CONT_NODE   *pLeft;     /* Left most contour node traced for the segment */
    ADI_CONT_HDR    *pDown;     /* next contour piece(if any) of the same segment */
};

/* Object type for Freeman contour header */
typedef struct _FreemanContourHdr   ADI_CONT_FREEMAN_HDR;
struct _FreemanContourHdr
{
    ADI_CONT_NODE   *pAnchor;   /* Pointer to contour */
    uint16_t        xOrigin;    /* x co-ordinate of contour origin */
    uint16_t        yOrigin;    /* y co-ordinate of contour origin */
};

/* Object type to hold Spatial moments */
typedef struct _SpatialMoments
{
    int32_t m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
} ADI_CONT_MOMENTS_SPATIAL;

/* Object type to hold Central moments */
typedef struct _CentralMoments
{
    int32_t mu20, mu11, mu02, mu30, mu21, mu12, mu03;
} ADI_CONT_MOMENTS_CENTRAL;

/* Object type to hold Normalized central moments */
typedef struct _CentralMomentsNormalized
{
    int32_t nScaleOfMoments;                            /* scale value of Normalized central moments */
    int32_t nu20, nu11, nu02, nu30, nu21, nu12, nu03;
} ADI_CONT_MOMENTS_CENT_NORM;

/* Object type to hold Hu moments */
typedef struct _HuMoments
{
    int32_t h0, h1, h2, h3, h4, h5, h6;
} ADI_CONT_MOMENTS_HU;

/* Object type for point representation */
typedef struct adi_Point
{
    uint16_t    nX;
    uint16_t    nY;
} ADI_POINT;

/* Object type for image dimension */
typedef struct adi_ImageSize
{
    uint32_t    nWidth;
    uint32_t    nHeight;
} ADI_IMAGE_SIZE;

/* Object type for rectangle */
typedef struct adi_Rectangle
{
    uint16_t    nX;
    uint16_t    nY;
    uint16_t    nWidth;
    uint16_t    nHeight;
} ADI_RECTANGLE;

/* Object type for image features */
typedef struct adi_ImageHaarData
{
    uint32_t        *pIntegralImage;
    uint64_t        *pSquareIntegral;
    ADI_IMAGE_SIZE  oImageSize;
} ADI_IMAGE_HAAR_DATA;

/* Object type for rectangle features */
typedef struct adi_HaarRectangle
{
    ADI_RECTANGLE   nRect;
    int32_t         nWeight;
} ADI_HAARRECTANGLE;

/* Object type for haar features */
typedef struct ADI_HaarFeature
{
    int32_t             nTilted;
    ADI_HAARRECTANGLE   nHaarRectangle[3];
} ADI_HAARFEATURE;

/* Object type for haar classifier */
typedef struct ADI_HaarClassifier
{
    int32_t         nCount;
    ADI_HAARFEATURE *pHaarFeature;
    int32_t         *pThreshold;
    int32_t         *pLeft;
    int32_t         *pRight;
    int32_t         *pAlpha;
} ADI_HAARCLASSIFIER;

/* Object type for haar stage classifier */
typedef struct ADI_HaarStageClassifier
{
    int32_t             nCount;
    int32_t             nThreshold;
    ADI_HAARCLASSIFIER  *pClassifier;
} ADI_HAARSTAGECLASSIFIER;

/* Object type for internal rectangle feature */
typedef struct adi_PvtHaarRectangle
{
    uint32_t    *p0, *p1, *p2, *p3;
    int32_t     nWeight;
} ADI_PVT_HAARRECTANGLE;

/* Object type for internal haar feature */
typedef struct adi_Pvt_HaarFeature
{
    ADI_PVT_HAARRECTANGLE   oPvtHaarRect[3];
} ADI_PVT_HAARFEATURE;

/* Object type for internal haar node */
typedef struct adi_PvtHaarTreeNode
{
    ADI_PVT_HAARFEATURE oFeature;
    int32_t             nThreshold;
    int32_t             nLeft;
    int32_t             nRight;
} ADI_PVT_HAARTREENODE;

/* Object type for internal haar classifier */
typedef struct adi_PvtHaarClassifier
{
    int32_t                 nCount;
    ADI_PVT_HAARTREENODE    *pNode;
    int32_t                 *pAlpha;
} ADI_PVT_HAARCLASSIFIER;

/* Object type for internal haar stage classifier */
typedef struct adi_PvtHaarStageClassifier
{
    int32_t                 nCount;
    int32_t                 nThreshold;
    ADI_PVT_HAARCLASSIFIER  *pClassifier;
    int32_t                 nTwoRects;
} ADI_PVT_HAARSTAGECLASSIFIER;

/* Object type for internal cascade classifier */
typedef struct adi_PvtClassifierCascade
{
    int32_t                     nCount;
    int32_t                     nInverseWindowArea;
    uint32_t                    *pSum;
    uint64_t                    *pSqSum;
    ADI_PVT_HAARSTAGECLASSIFIER *pStageClassifier;
    uint64_t                    *pq0, *pq1, *pq2, *pq3;
    uint32_t                    *p0, *p1, *p2, *p3;
} ADI_PVT_CLASSIFIERCASCADE;

/* Object type for haar cascade classifier */
typedef struct adi_HaarClassifierCascade
{
    int32_t                     nCount;
    ADI_IMAGE_SIZE              oOriginalWindowSize;
    ADI_IMAGE_SIZE              oRealWindowSize;
    float32_t                   nScale;
    ADI_HAARSTAGECLASSIFIER     *pStageClassifier;
    ADI_PVT_CLASSIFIERCASCADE   *pPvtCascade;
} ADI_HAARCLASSIFIERCASCADE;

/* Object type to hold 8bit Run-Length encoded node */
typedef struct _RunLenNode8bit
{

    /* nRun before nVal => little-endian mode packets */
    uint8_t nRun;   /* number of occurrences of pixel */
    uint8_t nVal;   /* gray-scale value of pixel */
} ADI_RUN_LEN_NODE_8BIT;

/* Object type to hold 16bit Run-Length encoded node */
typedef struct _RunLenNode16bit
{

    /* nRun before nVal => little-endian mode packets */
    uint16_t    nRun;   /* number of occurrences of pixel */
    int16_t     nVal;   /* gray-scale value of pixel */
} ADI_RUN_LEN_NODE_16BIT;

/* Object type to hold 32bit Run-Length encoded node */
typedef struct _RunLenNode32bit
{

    /* nRun before nVal => little-endian mode packets */
    uint32_t    nRun;   /* number of occurrences of pixel */
    uint32_t    nVal;   /* gray-scale value of pixel */
} ADI_RUN_LEN_NODE_32BIT;

typedef struct _adi_packed_rle  ADI_PACKED_RLE;
struct _adi_packed_rle
{
    ADI_RUN_LEN_NODE_16BIT  oStateOfLastRLENode;

    ADI_ITB_MODULE_STATUS   (*pfPACKED_RLE)(ADI_PACKED_RLE          *pRLEData,
                                            const int32_t           *pData,
                                            uint32_t                nNumData,
                                            ADI_RUN_LEN_NODE_16BIT  *pRunLenNodeArray,
                                            uint32_t                nMaxCount,
                                            uint32_t                *pnNumUsedRLENodes);

    void                    (*pfPACKED_RLE_reset)(ADI_PACKED_RLE            *pRLEData,
                                                  ADI_RUN_LEN_NODE_16BIT    *pRunLenNodeArray,
                                                  uint32_t                  nLastRLEIndex);
};

/* ADI_UKF will hold state and covariance data for current and predicted states */
typedef struct _adiUKF
{
    const uint32_t  nStates;
    const uint32_t  nMeasurements;
    const uint32_t  nScale;
    const uint32_t  nWmScale;
    uint32_t        nMeans;

    int32_t         *P;         /* initial state covraiance */
    int32_t         *Q;         /* covariance of process */
    int32_t         *R;         /* covariance of measurement */

    int32_t         c;          /* coefficient */
    int32_t         *x;         /* state estimate */
    int32_t         *z;         /* current measurement */

    int32_t         *Wm;        /* weights for mean */
    int32_t         *Wc;        /* weights for covraiance */

    int8_t          *pTempMem;  /* Temporary Space for intermediate computations */
} ADI_UKF;

/*=============  E X T E R N A L S  ============*/

/*
**  External Data section
*/

/*
**  External Function Prototypes
*/
//#else /*  assembly language specific macros and extern declarations. */
//#endif /* if !defined(_LANGUAGE_C) */

#ifdef _MISRA_RULES
#pragma diag(pop)
#endif
#endif /* __ADI_IMAGE_TOOL_BOX_DEFINES_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/tags/release_1_4_0_imagetoolbox_111510/include/adi_image_tool_box_defines.h $
**
*/

