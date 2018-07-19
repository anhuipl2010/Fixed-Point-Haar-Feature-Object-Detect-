/*****************************************************************************
Copyright (c) 2009 - Analog Devices Inc. All Rights Reserved.
This software is proprietary & confidential to Analog Devices, Inc.
and its licensors.
******************************************************************************
$Revision: 1380 $
$Date: 2009-11-02 17:03:10 +0530 (Mon, 02 Nov 2009) $

Title       : adi_haarfeatures.h

Description : declarations specific to Hough Transform modules

*****************************************************************************/
#ifndef __ADI_HAARFEATURES_H__
#define __ADI_HAARFEATURES_H__

/*=============  I N C L U D E S   =============*/

/*==============  D E F I N E S  ===============*/
#define ADI_IMAGE_WIDTH                 ((uint32_t) 320)
#define ADI_IMAGE_HEIGHT                ((uint32_t) 240)
#define ADI_ROI_IMAGE_WIDTH             ((uint32_t) 320)
#define ADI_ROI_IMAGE_HEIGHT            ((uint32_t) 240)
//#define ADI_MAX_FACE_WIDTH              20
//#define ADI_MAX_FACE_HEIGHT             20
#define ADI_MAX_HAAR_RECTANGLE_FEATURES 3


#define NUM_INPUT_FILES             16800//1845	//135
#define BLOCK_SIZE_L3               ((uint32_t) (ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT * ADI_HALFWORD_BYTES))
#define PPM_HEADER_SIZE             100
#define ADI_MEMORYFOR_TRAINEDDATA   55960	//264808	//129224	//70856	//886688	//583764
#define ADI_TRAINED_FILE_SIZE       16092	//76548	//37300	//20404	//256548

#define ADI_MIN_OBJECTWIDTH         24//12	//20	//12	//24		//6
#define ADI_MIN_OBJECTHEIGHT        12//8	//20	//12	//24		//6
#define ADI_MAX_FACE_WIDTH          120		//24	//60
#define ADI_MAX_FACE_HEIGHT         60	//240		//24	//60
#define ADI_MAX_FACES               500	// If this value is too small, the program is easily to collapse

#define ADI_MAX_HAAR_RECTANGLE_FEATURES 3
#define ADI_MAX_VALUE_131				2147483647
#define ADI_ONE_IN1616                  0x00010000

#define ASM_MUL16(nVal1, nVal2)     (((int16_t) nVal1 * (int16_t) nVal2) << 1)
#define UBROUND2POS(nValue, nPos)   (nValue + (1 << (nPos - 1)) - 1)

enum COMMAND
{
	MACRO_HAARFEATURE_FACEDETECT
};

/*=============  E X T E R N A L S  ============*/

/*
**  External Data section
*/

/*
**  External Function Prototypes
*/
uint32_t    adi_CreateInternalStructure(
                                        ADI_HAARCLASSIFIERCASCADE   *cascade,
                                        ADI_PVT_CLASSIFIERCASCADE   *pInternalStructPtr,
                                        uint32_t                    nBytesPassed
                                        );

uint32_t    adi_HaarFeaturesInit(
                                 ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
                                 uint32_t                    nSizeOfMemoryPassed,
                                 int8_t                      *pTrainedFileData,
                                 uint32_t                    nTrainedDataSize
                                 );

void        adi_HaarPreProcess(
                               ADI_IMAGE_HAAR_DATA *pImageData,
                               uint8_t             *pImage,
                               uint16_t            *pSquareImage,
                               uint32_t            *pIntegralImage,
                               uint64_t            *pSquareIntegral,
                               uint32_t            *pTempMemory,
                               uint32_t            nImageWidth,
                               uint32_t            nImageHeight,
                               uint32_t            nStride
                               );

uint32_t    adi_HaarPostProcess(
                                uint16_t    *pDetectedFaces,
                                uint16_t    *pTempMemory,
                                uint32_t    nNumDetectedFaces,
                                uint32_t    nPercentageOvelap,
                                uint32_t    nMinNeighbours
                                );

uint32_t    adi_HaarDetectObjects(
                                  ADI_IMAGE_HAAR_DATA         *pImageData,
                                  float32_t                   nScaleIncrement,
                                  ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
                                  ADI_IMAGE_SIZE              *pMinSizeObject,
                                  uint16_t                    *pDetectedFaces,
                                  uint8_t                     *pTempPtr
                                  );

uint32_t    adi_CreateHaarClassifierCascade(
                                            ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
                                            uint32_t                    nSizeOfMemoryPassed,
                                            int8_t                      *pTrainedData,
                                            uint32_t                    nTrainednTotalStructureMemory
                                            );

float32_t   adi_FindMax(
                        float32_t   nNum1,
                        float32_t   nNum2
                        );

int32_t     adi_Round(
                      float32_t   nVal
                      );

//void        adi_IntegralSquareSum_i16(
//	const uint16_t  *pInBuff,
//	uint64_t        *pSqSum,
//	uint32_t        dimY,
//	uint32_t        dimX,
//	uint32_t        nStride,
//	uint64_t        *pRowVector,
//	uint64_t        *pColVector
//	);

void adi_IntegralSum_i8(
	const uint8_t   *pInBuff,
	uint32_t        *pSum,
	uint32_t        dimY,
	uint32_t        dimX,
	uint32_t        nStride,
	uint32_t        *pRowVector,
	uint32_t        *pColVector
	);

void adi_IntegralSquareSum_i8(
	const uint16_t  *pInBuff,
	uint64_t        *pSqSum,
	uint32_t        dimY,
	uint32_t        dimX,
	uint32_t        nStride,
	uint64_t        *pRowVector,
	uint64_t        *pColVector
	);

void adi_ReScaleFeaturesForSubWindow(
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA   *pImageData,
	int32_t                     nScale
	);

int32_t adi_EvaluateSubWindow(
	const ADI_HAARCLASSIFIERCASCADE *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA       *pImageData,
	ADI_POINT                       oStartPoint,
	int32_t                         nStartStage
	);

void adi_RGB2GRAY(
	unsigned char *src,
	unsigned char *dest,
	int width,
	int height);

int32_t adi_mult1616_32bit_o16(
	int32_t nNum1,
	int32_t nNum2
	);

int32_t adi_mult1616_131(
	int32_t nNum1,
	int32_t nNum2
	);

int32_t adi_mult1616_32bit(
	int32_t nNum1,
	int32_t nNum2
	);

void adi_ReScaleFeaturesForSubWindow(
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA   *pImageData,
	int32_t                     nScale
	);

int32_t adi_mult131_32bit(
	int32_t nNum1,
	int32_t nNum2
	);

int16_t _sqrt_fr16(int16_t  nInp);

uint32_t adi_SqrRootFixed(
	uint32_t    nNum,
	uint32_t    nShift
	);

int32_t adi_mult1616_1616(
	int32_t nNum1,
	int32_t nNum2
	);

int32_t adi_EvaluateSubWindow(
	const ADI_HAARCLASSIFIERCASCADE *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA       *pImageData,
	ADI_POINT                       oStartPoint,
	int32_t                         nStartStage
	);

extern int haar_classifir_cascade[];
extern int stage_classifier_count[];
extern int stage_classifier_threshold[];
extern int classifier_node_count[];
extern int classifier_node_threshold[];
extern int left_values[];
extern int right_values[];
extern int alpha[];
extern int tilted[];
extern int feature_rect[];
extern int weight[];
#define RECT_NUM (3)
/*=============  D A T A    T Y P E S   =============*/
#endif /* __ADI_HOUGH_TRANSFORM_H__ */

/*
**
** EOF: $URL: http://ipdc-autosrv1.ad.analog.com:8080/svn/ipdc-ptg/trunk/image_processing_toolbox/demo/include/adi_hough_transform.h $
**
*/
