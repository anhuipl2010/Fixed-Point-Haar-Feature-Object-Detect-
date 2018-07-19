#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adi_image_tool_box.h"
#include "adi_haarfeatures.h"
//#include "adi_util_funcs.h"
#include "profile.h"
#include <math.h>

static int16_t  sqrtcoef0[5] =
{
	/*
	0x2D41, 0xD2CE, 0xE7E8, 0xF848, 0xAC7C
	*/
	11585, -11570, -6168, -1976, -21380
};

static int16_t  sqrtcoef1[5] =
{
	/*
	0x2D42, 0x2D31, 0xEA5D, 0x1021, 0xF89E
	*/
	11586, 11569, -5539, 4129, -1890
};

void adi_RGB2GRAY(unsigned char *src,unsigned char *dest,int width,int height)
{
	if (src == NULL || dest == NULL)
	{
		return;
	}

	int r, g, b;
	for (int i=0;i<width*height;++i)
	{
		b = *src++; // load red
		g = *src++; // load green
		r = *src++; // load blue
		// build weighted average:
		*dest++ = (r * 76 + g * 150 + b * 30) >>8;
		//printf("gray value = %d\n", (r * 76 + g * 150 + b * 30) >>8);
	}
}

uint32_t adi_HaarFeaturesInit(
	ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
	uint32_t                    nSizeOfMemoryPassed,
	int8_t                      *pTrainedFileData,
	uint32_t                    nTrainedDataSize
	)
{
	int32_t                     nBytesUsed;
	int32_t                     nBytesLeft;
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier;
	ADI_PVT_CLASSIFIERCASCADE   *pInternalCascade;
	nBytesLeft = nSizeOfMemoryPassed;
	pCascadeClassifier = pClassifierCascade;
	nBytesLeft = adi_CreateHaarClassifierCascade(pCascadeClassifier,
		nBytesLeft,
		pTrainedFileData,
		nTrainedDataSize);
	nBytesUsed = nSizeOfMemoryPassed - nBytesLeft;
	pInternalCascade = (ADI_PVT_CLASSIFIERCASCADE *) ((uint8_t *)pCascadeClassifier + nBytesUsed);
	if (!pCascadeClassifier->pPvtCascade)
	{
		nBytesLeft = adi_CreateInternalStructure(pCascadeClassifier,
			pInternalCascade,
			nBytesLeft);
	}

	return (nBytesLeft);
}

void adi_HaarPreProcess(
	ADI_IMAGE_HAAR_DATA *pImageData,
	uint8_t             *pImage,
	uint16_t            *pSquareImage,
	uint32_t            *pIntegralImage,
	uint64_t            *pSquareIntegral,
	uint32_t            *pTempMemory,
	uint32_t            nImageWidth,
	uint32_t            nImageHeight,
	uint32_t            nStride
	)
{
	uint8_t     *pInputImage;
	uint32_t    nValue;
	uint32_t    i;

	pInputImage = pImage;
	pImageData->oImageSize.nWidth = nImageWidth;
	pImageData->oImageSize.nHeight = nImageHeight;
	memset(pIntegralImage, 0, nImageWidth * nImageHeight * 4);
	memset(pSquareIntegral, 0, nImageWidth * nImageHeight * 8);
	memset(pTempMemory, 0, nImageWidth * nImageHeight);
	adi_IntegralSum_i8(pImage,
		pIntegralImage,
		nImageHeight,
		nImageWidth,
		nStride,
		pTempMemory,
		pTempMemory);
	for (i = 0; i < (nImageWidth * nImageHeight); i++)
	{
		nValue = pImage[i];
		pSquareImage[i] = nValue * nValue;
	}

	adi_IntegralSquareSum_i8(pSquareImage,
		pSquareIntegral,
		nImageHeight,
		nImageWidth,
		nStride,
		(uint64_t *)pTempMemory,
		(uint64_t *)pTempMemory);
	pImageData->pIntegralImage = pIntegralImage;
	pImageData->pSquareIntegral = pSquareIntegral;
}

uint32_t adi_HaarDetectObjects(
	ADI_IMAGE_HAAR_DATA         *pImageData,
	float32_t                   nScaleIncrement,
	ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
	ADI_IMAGE_SIZE              *pMinSizeObject,
	uint16_t                    *pDetectedFaces,
	uint8_t                     *pTempPtr
	)
{
	uint32_t                    nSplitStage;
	float32_t                   nFactor;
	int32_t                     nPass;
	float32_t                   nYStep;
	uint32_t                    nXStep;
	ADI_IMAGE_SIZE              oWindowSize;
	ADI_RECTANGLE               nEquRectangle = { 0, 0, 0, 0 };
	int32_t                     *p0, *p1 = 0, *p2 = 0, *p3 = 0;
	int32_t                     *pq0, *pq1 = 0, *pq2 = 0, *pq3 = 0;
	int32_t                     nPassCount, nStageOffset;
	int32_t                     nStopHeight, nStopWidth;
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier;
	uint16_t                    *pDetectedLikeFace;
	uint32_t                    nCountX, nCountY;
	uint32_t                    nRescaleX, nRescaleY;
	int32_t                     nResult;
	uint8_t                     *pMaskRow;
	uint32_t                    nBasicCount;	//This value can cause an array to cross the boundary, it may cause the program collapse
	float32_t                   nFactorStart;
	ADI_POINT                   oPoint;
	uint32_t                    nSubwindowScale;
#ifdef PROFILE
	uint32_t                    nNumSubWindows = 0;
	uint32_t                    nNumSizeWindows = 0;
#endif
	nSplitStage = 20;	//2
	nPass = 2;
	p0 = 0;
	pq0 = 0;
	nStageOffset = 0;
	nBasicCount = 0;
	pCascadeClassifier = pClassifierCascade;
	pDetectedLikeFace = pDetectedFaces;
	pMaskRow = pTempPtr;
#ifdef PROFILE
	goProfileFunc.nSum = 0;
	goProfileFunc1.nSum = 0;
#endif
	PROFBEG(goProfile);
	if (nSplitStage >= (uint32_t)(pCascadeClassifier->nCount))
	{
		nSplitStage = pCascadeClassifier->nCount;
		nPass = 1;
	}

	//nFactorStart = (float32_t) pMinSizeObject->nWidth / pCascadeClassifier->oOriginalWindowSize.nWidth;
	nFactorStart = pow(1.4567, 4);
	for
		(
		nFactor = nFactorStart;
	(nFactor * pCascadeClassifier->oOriginalWindowSize.nWidth) < ADI_MAX_FACE_WIDTH
		&&  (nFactor * pCascadeClassifier->oOriginalWindowSize.nHeight) < ADI_MAX_FACE_HEIGHT;
	nFactor *= nScaleIncrement
		)
	{
		nYStep = adi_FindMax(2, nFactor);
		oWindowSize.nWidth = adi_Round(pCascadeClassifier->oOriginalWindowSize.nWidth * nFactor);
		oWindowSize.nHeight = adi_Round(pCascadeClassifier->oOriginalWindowSize.nHeight * nFactor);
		nStopHeight = adi_Round((pImageData->oImageSize.nHeight - oWindowSize.nHeight) / nYStep);
#ifdef PROFILE        
		nNumSizeWindows++;
#endif        
		PROFBEG(goProfileFunc);
		nSubwindowScale = adi_Round(nFactor * 65536);
		printf("width = %d, height = %d, factor = %d\n", oWindowSize.nWidth, oWindowSize.nHeight, nSubwindowScale);//add by ming jun 2018-07-12
		adi_ReScaleFeaturesForSubWindow(pCascadeClassifier,
			pImageData,
			nSubwindowScale);
		PROFEND(goProfileFunc);
		pCascadeClassifier->pPvtCascade->nCount = nSplitStage;
		for (nPassCount = 0; nPassCount < nPass; nPassCount++)
		{
			for (nCountY = 0; nCountY < (uint32_t)nStopHeight; nCountY++)
			{
				nRescaleY = adi_Round(nCountY * nYStep);
				nXStep = 1;

				nStopWidth = adi_Round((pImageData->oImageSize.nWidth - oWindowSize.nWidth) / nYStep);

				/*Please Check This First.Potential Bug */
				pMaskRow = pTempPtr + nRescaleY * pImageData->oImageSize.nWidth;
				for (nCountX = 0; nCountX < (uint32_t)nStopWidth; nCountX += nXStep)
				{
					nRescaleX = adi_Round(nCountX * nYStep);
					if (nPassCount == 0)
					{
						nXStep = 2;
						oPoint.nX = nRescaleX;
						oPoint.nY = nRescaleY;
#ifdef PROFILE                          
						nNumSubWindows++;
#endif                        
						PROFBEG(goProfileFunc1);
						nResult = adi_EvaluateSubWindow(pCascadeClassifier,
							pImageData,
							oPoint,
							0);
						PROFEND(goProfileFunc1);
						if (nResult > 0)
						{
							if (nPassCount < nPass - 1)
							{
								pMaskRow[nRescaleX] = 1;
							}
							else
							{
								*pDetectedLikeFace++ = nRescaleX;
								*pDetectedLikeFace++ = nRescaleY;
								*pDetectedLikeFace++ = oWindowSize.nWidth;
								*pDetectedLikeFace++ = oWindowSize.nHeight;
								nBasicCount++;

								//add by ming jun 2018-07-16
								if (nBasicCount == ADI_MAX_FACES)
								{
									return (nBasicCount);
								}
								//end
							}
						}

						if (nResult < 0)
						{
							nXStep = 1;
						}
					}
					else if (pMaskRow[nRescaleX])
					{
						oPoint.nX = nRescaleX;
						oPoint.nY = nRescaleY;
#ifdef PROFILE                          
						nNumSubWindows++;
#endif
						PROFBEG(goProfileFunc1);
						nResult = adi_EvaluateSubWindow(pCascadeClassifier,
							pImageData,
							oPoint,
							nStageOffset);
						PROFEND(goProfileFunc1);
						if (nResult > 0)
						{
							if (nPassCount == nPass - 1)
							{
								*pDetectedLikeFace++ = nRescaleX;
								*pDetectedLikeFace++ = nRescaleY;
								*pDetectedLikeFace++ = oWindowSize.nWidth;
								*pDetectedLikeFace++ = oWindowSize.nHeight;
								nBasicCount++;

								//add by ming jun 2018-07-16
								if (nBasicCount == ADI_MAX_FACES)
								{
									return (nBasicCount);
								}
								//end
							}
						}
						else
						{
							pMaskRow[nRescaleX] = 0;
						}
					}
				}
			}

			nStageOffset = pCascadeClassifier->pPvtCascade->nCount;
			pCascadeClassifier->pPvtCascade->nCount = pCascadeClassifier->nCount;
		}
	}

#ifdef PROFILE
	PRINT("Only Func Cycles(Rescale)/subwindow = %f\n",
		(float32_t) ((float64_t) goProfileFunc.nSum/nNumSizeWindows));
	PRINT("Only Func Cycles(EvaluateSubWindow)/subwindow = %f\n",
		(float32_t) ((float64_t) goProfileFunc1.nSum/nNumSubWindows));
#endif
	return (nBasicCount);
}

uint32_t adi_HaarPostProcess(
	uint16_t    *pDetectedFaces,
	uint16_t    *pTempMemory,
	uint32_t    nNumDetectedFaces,
	uint32_t    nPercentageOvelap,
	uint32_t    nMinNeighbours
	)
{
	uint32_t    i, j;
	uint16_t    *pArrangedFaces;
	uint16_t    *pMergedFaces;
	uint16_t    *pRawFaces;
	uint16_t    nX, nY;
	uint32_t    nRectWidth, nRectHeight;
	uint32_t    nCount;
	int32_t     nXLeft, nXRight, nYLeft, nYRight;
	uint32_t    nNeighbourFaces;
	uint32_t    nTotalProcessedFaces;
	uint32_t    nFaceGroups;
	uint32_t    nFlag;
	uint32_t    nActualFacesDetected;
	uint32_t    nDistance;
	uint32_t    nXOverlap, nYOverLap;

	nCount = 0;
	nFaceGroups = 0;
	nFlag = 0;
	nActualFacesDetected = 0;

	pArrangedFaces = pTempMemory;
	pRawFaces = pDetectedFaces;
	nNeighbourFaces = 0;
	nTotalProcessedFaces = 0;
	for (i = 0; i < nNumDetectedFaces; i++)
	{
		*pArrangedFaces++ = 1;
		*pArrangedFaces++ = *pRawFaces++;
		*pArrangedFaces++ = *pRawFaces++;
		*pArrangedFaces++ = *pRawFaces++;
		*pArrangedFaces++ = *pRawFaces++;
	}

	pArrangedFaces = pTempMemory;
	for (i = 0; i < (nNumDetectedFaces); i++)
	{
		if (pArrangedFaces[i * 5] > 0)
		{
			nX = pArrangedFaces[(i * 5) + 1];
			nY = pArrangedFaces[(i * 5) + 2];
			nRectWidth = pArrangedFaces[(i * 5) + 3];
			nRectHeight = pArrangedFaces[(i * 5) + 4];
			nXOverlap = nRectWidth * nPercentageOvelap / 100;
			nYOverLap = nRectHeight * nPercentageOvelap / 100;
			nXLeft = nX - nXOverlap;
			nXRight = nX + nXOverlap;
			nYLeft = nY - nYOverLap;
			nYRight = nY + nYOverLap;
			pMergedFaces = pTempMemory;
			for (j = (i + 1); j < nNumDetectedFaces; j++)
			{
				nFlag = 0;
				if (pMergedFaces[j * 5] > 0)
				{
					if
						(
						nRectWidth == pMergedFaces[j * 5 + 3]
					&&  nRectHeight == pMergedFaces[j * 5 + 4]
					&&  nXLeft < pMergedFaces[j * 5 + 1]
					&&  nXRight > pMergedFaces[j * 5 + 1]
					&&  nYLeft < pMergedFaces[j * 5 + 2]
					&&  nYRight > pMergedFaces[j * 5 + 2]
					)
					{
						nCount = pArrangedFaces[(i * 5)];
						nCount++;
						pArrangedFaces[(i * 5)] = nCount;
						pMergedFaces[j * 5 + 0] = 0;
						pArrangedFaces[(i * 5) + 1] += pMergedFaces[j * 5 + 1];
						pArrangedFaces[(i * 5) + 2] += pMergedFaces[j * 5 + 2];
						nFlag = 1;
					}
				}
			}
		}
	}

	nFaceGroups = 0;
	for (i = 0; i < nNumDetectedFaces; i++)
	{
		if (*pArrangedFaces > 0)
		{
			nFaceGroups++;
			nNeighbourFaces = *pArrangedFaces++;
			*pMergedFaces++ = nNeighbourFaces;
			nX = *pArrangedFaces++;
			nX = (nX * 2 + nNeighbourFaces) / (2 * nNeighbourFaces);
			nY = *pArrangedFaces++;
			nY = (nY * 2 + nNeighbourFaces) / (2 * nNeighbourFaces);
			*pMergedFaces++ = nX;
			*pMergedFaces++ = nY;
			*pMergedFaces++ = *pArrangedFaces++;
			*pMergedFaces++ = *pArrangedFaces++;
		}
		else
		{
			pArrangedFaces = pArrangedFaces + 5;
		}
	}

	pArrangedFaces = pTempMemory;
	pMergedFaces = pTempMemory;

	/* Removing small face rectangles inside large face rectangles */
	for (i = 0; i < nFaceGroups; i++)
	{
		nFlag = 1;
		for (j = 0; j < nFaceGroups; j++)
		{
			nDistance = adi_Round(pMergedFaces[j * 5 + 3] * (float32_t)0.2);
			if(
				i != (uint32_t)j &&
				pArrangedFaces[i * 5 + 1] >= (pMergedFaces[j * 5 + 1] - nDistance) &&
				pArrangedFaces[i * 5 + 2] >= (pMergedFaces[j * 5 + 2] - nDistance) &&
				(pArrangedFaces[i * 5 + 1] + pArrangedFaces[i * 5 + 3]) <= (pMergedFaces[j * 5 + 1] + pMergedFaces[j * 5 + 3] + (uint16_t)nDistance) &&
				(pArrangedFaces[i * 5 + 2] + pArrangedFaces[i * 5 + 4]) <= pMergedFaces[j * 5 + 2] + pMergedFaces[j * 5 + 4] + (uint16_t)nDistance
				)
			{
				nFlag = 0;
				break;
			}
		}

		if (nFlag)
		{
			if (pArrangedFaces[i * 5] > nMinNeighbours)
			{
				nActualFacesDetected++;
				*pDetectedFaces++ = pArrangedFaces[i * 5 + 1];
				*pDetectedFaces++ = pArrangedFaces[i * 5 + 2];
				*pDetectedFaces++ = pArrangedFaces[i * 5 + 3];
				*pDetectedFaces++ = pArrangedFaces[i * 5 + 4];
			}
		}
	}

	return (nActualFacesDetected);
}

int32_t adi_Round(float32_t   nVal)
{
	int32_t nResult;
	nVal = nVal + (float32_t)0.5;
	nResult = (int32_t) nVal;
	return (nResult);
}

uint32_t adi_CreateInternalStructure(
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier,
	ADI_PVT_CLASSIFIERCASCADE   *pInternalStructPtr,
	uint32_t                    nBytesPassed
	)
{
	ADI_PVT_CLASSIFIERCASCADE   *pPvtCascadeClassifier;
	int32_t                     i, j, k, l;
	int32_t                     nTotalSize;
	int32_t                     nTotalClassifiers;
	int32_t                     nTotalNodes;
	ADI_PVT_HAARCLASSIFIER      *pPvtHaarClassifier;
	ADI_PVT_HAARTREENODE        *pPvtNode;
	ADI_HAARCLASSIFIER          *pClassifier;
	ADI_IMAGE_SIZE              oOriginalWindowSize;
	ADI_HAARSTAGECLASSIFIER     *pStageClassifier;
	ADI_PVT_HAARSTAGECLASSIFIER *pPvtStageClassifier;
	ADI_RECTANGLE               oRect;
	int32_t                     nTilted;
	int32_t                     nMaxCount;
	int32_t                     nNodeCount;
	int32_t                     *pAlphaPtr;
	ADI_PVT_HAARTREENODE        *pPvtTreeNode;
	ADI_HAARFEATURE             *pFeature;
	ADI_PVT_HAARCLASSIFIER      *pPvtClassifier;

	pPvtCascadeClassifier = 0;
	nTotalClassifiers = 0;
	nTotalNodes = 0;
	nMaxCount = 0;

	oOriginalWindowSize = pCascadeClassifier->oOriginalWindowSize;
	pStageClassifier = pCascadeClassifier->pStageClassifier;

	/* check input structure correctness and calculate total memory size needed for
	internal representation of the classifier cascade */
	for (i = 0; i < pCascadeClassifier->nCount; i++)
	{
		pStageClassifier = pCascadeClassifier->pStageClassifier + i;
		nMaxCount = (int32_t)adi_FindMax((float32_t)nMaxCount, (float32_t)pStageClassifier->nCount);
		nTotalClassifiers += pStageClassifier->nCount;
		for (j = 0; j < pStageClassifier->nCount; j++)
		{
			pClassifier = pStageClassifier->pClassifier + j;
			nTotalNodes += pClassifier->nCount;
			for (l = 0; l < pClassifier->nCount; l++)
			{
				for (k = 0; k < ADI_MAX_HAAR_RECTANGLE_FEATURES; k++)
				{
					if
						(
						pClassifier->pHaarFeature[l].nHaarRectangle[k].nRect.nWidth
						)
					{
						oRect = pClassifier->pHaarFeature[l].nHaarRectangle[k].nRect;
						nTilted = pClassifier->pHaarFeature[l].nTilted;
					}
				}
			}
		}
	}

	nTotalSize = sizeof(ADI_PVT_CLASSIFIERCASCADE)
		+ sizeof(ADI_PVT_HAARSTAGECLASSIFIER) * pCascadeClassifier->nCount
		+ sizeof(ADI_PVT_HAARCLASSIFIER) * nTotalClassifiers
		+ sizeof(ADI_PVT_HAARTREENODE) * nTotalNodes
		+ sizeof(void *) * (nTotalNodes + nTotalClassifiers);
	if (nBytesPassed < (uint32_t)nTotalSize)
	{
		return (nBytesPassed - nTotalSize);
	}

	pPvtCascadeClassifier = (ADI_PVT_CLASSIFIERCASCADE *)pInternalStructPtr;
	memset(pPvtCascadeClassifier, 0, sizeof(*pPvtCascadeClassifier));

	/* init header */
	pPvtCascadeClassifier->nCount = pCascadeClassifier->nCount;
	pPvtCascadeClassifier->pStageClassifier = (ADI_PVT_HAARSTAGECLASSIFIER *) (pPvtCascadeClassifier + 1);
	pPvtHaarClassifier = (ADI_PVT_HAARCLASSIFIER *) (pPvtCascadeClassifier->pStageClassifier + pCascadeClassifier->nCount);
	pPvtNode = (ADI_PVT_HAARTREENODE *) (pPvtHaarClassifier + nTotalClassifiers);

	/* initialize internal representation */
	for (i = 0; i < pCascadeClassifier->nCount; i++)
	{
		pStageClassifier = pCascadeClassifier->pStageClassifier + i;
		pPvtStageClassifier = pPvtCascadeClassifier->pStageClassifier + i;

		pPvtStageClassifier->nCount = pStageClassifier->nCount;
		pPvtStageClassifier->nThreshold = pStageClassifier->nThreshold;
		pPvtStageClassifier->pClassifier = pPvtHaarClassifier;
		pPvtStageClassifier->nTwoRects = 1;
		pPvtHaarClassifier += pStageClassifier->nCount;
		for (j = 0; j < pStageClassifier->nCount; j++)
		{
			pClassifier = pStageClassifier->pClassifier + j;
			pPvtClassifier = pPvtStageClassifier->pClassifier + j;
			nNodeCount = pClassifier->nCount;
			pAlphaPtr = (int32_t *) (pPvtNode + nNodeCount);

			pPvtClassifier->nCount = nNodeCount;
			pPvtClassifier->pNode = pPvtNode;
			pPvtClassifier->pAlpha = pAlphaPtr;

			for (l = 0; l < nNodeCount; l++)
			{
				pPvtTreeNode = pPvtClassifier->pNode + l;
				pFeature = pClassifier->pHaarFeature + l;
				memset(pPvtTreeNode, -1, sizeof(*pPvtTreeNode));
				pPvtTreeNode->nThreshold = pClassifier->pThreshold[l];
				pPvtTreeNode->nLeft = pClassifier->pLeft[l];
				pPvtTreeNode->nRight = pClassifier->pRight[l];
			}

			memcpy(pAlphaPtr,
				pClassifier->pAlpha,
				(nNodeCount + 1) * sizeof(pAlphaPtr[0]));
			pPvtNode = (ADI_PVT_HAARTREENODE *) (pAlphaPtr + nNodeCount + 1);
		}
	}

	pCascadeClassifier->pPvtCascade = pPvtCascadeClassifier;
	return (nBytesPassed - nTotalSize);
}

uint32_t adi_CreateHaarClassifierCascade(
	ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade,
	uint32_t                    nSizeOfMemoryPassed,
	int8_t                      *pTrainedData,
	uint32_t                    nTrainednTotalStructureMemory
	)
{
	int32_t                 i, j, k, l;
	int32_t                 nTotalStructureMemory;
	int32_t                 nTotalCascade;
	int32_t                 nStageClassifiers;
	int32_t                 nNodes;
	int32_t                 nTotalNodes;
	int32_t                 nTotalStageClassifiers;
	ADI_HAARCLASSIFIER      *pHaarClassifier;
	int32_t                 *pThresholdPtr;
	int32_t                 *pLeftPtr;
	int32_t                 *pRightPtr;
	int32_t                 *pAlphaPtr;
	void                    *pTrainedDataPtr;
	int32_t                 *pInt32Pointer;
	uint8_t                 *pUint8Pointer;
	ADI_HAARSTAGECLASSIFIER *pStageClassifier;
	ADI_HAARFEATURE         *pHaarFeature;

	j = 0;
	nTotalCascade = 0;
	nStageClassifiers = 0;
	nNodes = 0;
	nTotalNodes = 0;
	nTotalStageClassifiers = 0;
	pTrainedDataPtr = pTrainedData;
	pInt32Pointer = (int32_t *)pTrainedDataPtr;
	nTotalCascade = *(pInt32Pointer++);
	pTrainedDataPtr = pInt32Pointer;
	for (i = 0; i < nTotalCascade; i++)
	{
		pInt32Pointer = (int32_t *)pTrainedDataPtr;
		nStageClassifiers = *(pInt32Pointer++);
		pTrainedDataPtr = pInt32Pointer;
		nTotalStageClassifiers += nStageClassifiers;
		for (j = 0; j < nStageClassifiers; j++)
		{
			pInt32Pointer = (int32_t *)pTrainedDataPtr;
			nNodes = *(pInt32Pointer++);
			pTrainedDataPtr = pInt32Pointer;
			nTotalNodes += nNodes;
			for (k = 0; k < nNodes; k++)
			{
				pUint8Pointer = (uint8_t *)pTrainedDataPtr;
				pUint8Pointer = pUint8Pointer + 84;
				pTrainedDataPtr = pUint8Pointer;
			}
		}

		pUint8Pointer = (uint8_t *)pTrainedDataPtr;
		pUint8Pointer = pUint8Pointer + 4;
		pTrainedDataPtr = pUint8Pointer;
	}

	nTotalStructureMemory = sizeof(ADI_HAARCLASSIFIERCASCADE)
		+ sizeof(ADI_HAARSTAGECLASSIFIER) * nTotalCascade
		+ sizeof(ADI_HAARCLASSIFIER) * nTotalStageClassifiers
		+ sizeof(ADI_HAARFEATURE) * nTotalNodes
		+ sizeof(int32_t *) * (nTotalNodes * 5);
	if ((uint32_t) nTotalStructureMemory > nSizeOfMemoryPassed)
	{
		return (nSizeOfMemoryPassed - nTotalStructureMemory);
	}

	memset(pClassifierCascade, 0, nTotalStructureMemory);
	pClassifierCascade->nCount = nTotalCascade;
	pClassifierCascade->oOriginalWindowSize.nHeight = ADI_MIN_OBJECTHEIGHT;	//24;
	pClassifierCascade->oOriginalWindowSize.nWidth = ADI_MIN_OBJECTWIDTH;	//24;
	pClassifierCascade->pStageClassifier = (ADI_HAARSTAGECLASSIFIER *) (pClassifierCascade + 1);

	pTrainedDataPtr = pTrainedData;
	pInt32Pointer = (int32_t *)pTrainedDataPtr;
	nTotalCascade = *(pInt32Pointer++);
	pTrainedDataPtr = pInt32Pointer;

	pStageClassifier = pClassifierCascade->pStageClassifier;
	pHaarClassifier = (ADI_HAARCLASSIFIER *) (pStageClassifier + nTotalCascade);
	pHaarFeature = (ADI_HAARFEATURE *) (pHaarClassifier + nTotalStageClassifiers);
	pThresholdPtr = (int32_t *) (pHaarFeature + nTotalNodes);
	pLeftPtr = (int32_t *) (pThresholdPtr + nTotalNodes);
	pRightPtr = (pLeftPtr + nTotalNodes);
	pAlphaPtr = (int32_t *) (pRightPtr + nTotalNodes);

	for (i = 0; i < nTotalCascade; i++)
	{
		pInt32Pointer = (int32_t *)pTrainedDataPtr;
		pStageClassifier->nCount = *(pInt32Pointer++);
		pTrainedDataPtr = pInt32Pointer;

		pStageClassifier->pClassifier = pHaarClassifier;
		for (j = 0; j < pStageClassifier->nCount; j++)
		{
			pInt32Pointer = (int32_t *)pTrainedDataPtr;
			pHaarClassifier->nCount = *(pInt32Pointer++);
			pTrainedDataPtr = pInt32Pointer;
			pHaarClassifier->pHaarFeature = pHaarFeature;
			for (k = 0; k < pHaarClassifier->nCount; k++)
			{
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				pHaarFeature->nTilted = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				for (l = 0; l < 3; l++)
				{
					pInt32Pointer = (int32_t *)pTrainedDataPtr;
					pHaarFeature->nHaarRectangle[l].nRect.nX = *(pInt32Pointer++);
					pTrainedDataPtr = pInt32Pointer;
					pInt32Pointer = (int32_t *)pTrainedDataPtr;
					pHaarFeature->nHaarRectangle[l].nRect.nY = *(pInt32Pointer++);
					pTrainedDataPtr = pInt32Pointer;
					pInt32Pointer = (int32_t *)pTrainedDataPtr;
					pHaarFeature->nHaarRectangle[l].nRect.nWidth = *(pInt32Pointer++);
					pTrainedDataPtr = pInt32Pointer;
					pInt32Pointer = (int32_t *)pTrainedDataPtr;
					pHaarFeature->nHaarRectangle[l].nRect.nHeight = *(pInt32Pointer++);
					pTrainedDataPtr = pInt32Pointer;
					pInt32Pointer = (int32_t *)pTrainedDataPtr;
					pHaarFeature->nHaarRectangle[l].nWeight = *(pInt32Pointer++);
					pTrainedDataPtr = pInt32Pointer;
				}

				pHaarClassifier->pThreshold = pThresholdPtr;
				pHaarClassifier->pLeft = pLeftPtr;
				pHaarClassifier->pRight = pRightPtr;
				pHaarClassifier->pAlpha = pAlphaPtr;
				pThresholdPtr++;
				pLeftPtr++;
				pRightPtr++;
				pAlphaPtr++;
				pAlphaPtr++;
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				*(pHaarClassifier->pThreshold) = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				*(pHaarClassifier->pLeft) = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				*(pHaarClassifier->pRight) = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				pHaarClassifier->pAlpha[0] = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				pInt32Pointer = (int32_t *)pTrainedDataPtr;
				pHaarClassifier->pAlpha[1] = *(pInt32Pointer++);
				pTrainedDataPtr = pInt32Pointer;
				pHaarFeature += 1;
			}

			pHaarClassifier += 1;
		}

		pInt32Pointer = (int32_t *)pTrainedDataPtr;
		pStageClassifier->nThreshold = *(pInt32Pointer++);
		pTrainedDataPtr = pInt32Pointer;
		pStageClassifier += 1;
	}

	return (nSizeOfMemoryPassed - nTotalStructureMemory);
}

float32_t adi_FindMax(
	float32_t   nNum1,
	float32_t   nNum2
	)
{
	float32_t   nResult;
	if (nNum1 > nNum2)
	{
		nResult = nNum1;
	}
	else
	{
		nResult = nNum2;
	}

	return (nResult);
}

void adi_IntegralSum_i8(
	const uint8_t   *pInBuff,
	uint32_t        *pSum,
	uint32_t        dimY,
	uint32_t        dimX,
	uint32_t        nStride,
	uint32_t        *pRowVector,
	uint32_t        *pColVector
	)
{
	uint32_t    i, j;
	uint8_t     *pIn;
	uint32_t    *pIntSum;
	uint32_t    *pRow;
	uint32_t    *pCol;
	int32_t     nResult;

	/* Input Buffer Pointer */
	pIn = (uint8_t *)pInBuff;

	/* Pointer to integral sum */
	pIntSum = pSum;

	/* Pointer to row vector */
	pRow = pRowVector;

	/* Pointer to column vector */
	pCol = pColVector;
	nResult = 0;

	for (i = 0; i < dimY; i++)
	{

		/* First column is separate because I(x-1,y)
		should be used from column vector */
		nResult = *pRow;
		nResult = nResult - *pCol++;
		nResult = nResult + *pCol;
		nResult = nResult + *pIn++;
		*pIntSum = nResult;
		for (j = 0; j < (dimX - 1); j++)
		{

			/* Middle and last elements of column */
			nResult = *pIn++;
			nResult = nResult + *pIntSum++;
			nResult = nResult - *pRow++;
			nResult = nResult + *pRow;
			*pIntSum = nResult;
		}

		pIntSum++;
		pIn = pIn + nStride;
		pRow = pIntSum - (dimX);
	}
}

void adi_IntegralSquareSum_i8(
	const uint16_t  *pInBuff,
	uint64_t        *pSqSum,
	uint32_t        dimY,
	uint32_t        dimX,
	uint32_t        nStride,
	uint64_t        *pRowVector,
	uint64_t        *pColVector
	)
{
	uint32_t    i, j;
	uint16_t    *pIn;
	uint64_t    *pIntSum;
	uint64_t    *pRow;
	uint64_t    *pCol;
	uint64_t    nResult;

	pIn = (uint16_t *)pInBuff;
	pIntSum = pSqSum;
	pRow = pRowVector;
	pCol = pColVector;
	nResult = 0;

	for (i = 0; i < dimY; i++)
	{

		/* First column is separate because I(x-1,y)
		should be used from column vector */
		nResult = *pRow;
		nResult = nResult + *pCol++;
		nResult = nResult + *pCol;
		nResult = nResult + *pIn++;
		*pIntSum = nResult;
		for (j = 0; j < (dimX - 1); j++)
		{

			/* Middle and last elements of column */
			nResult = *pIn++;
			nResult = nResult + *pIntSum++;
			nResult = nResult - *pRow++;
			nResult = nResult + *pRow;
			*pIntSum = nResult;
		}

		pIntSum++;
		pIn = pIn + nStride;
		pRow = pIntSum - (dimX);
	}
}

void adi_ReScaleFeaturesForSubWindow(
	ADI_HAARCLASSIFIERCASCADE   *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA   *pImageData,
	int32_t                     nScale
	)
{
	ADI_PVT_CLASSIFIERCASCADE   *pInternalClassifierCascade;
	uint16_t                    nPointX, nPointY;
	uint16_t                    nRectWidth, nRectHeight;
	int32_t                     nWeightScale;
	uint32_t                    *pIntegralImage;
	uint64_t                    *pSquareIntegralImage;
	int32_t                     nLoop1, nLoop2, nLoop3;
	ADI_HAARFEATURE             *pFeature;
	ADI_PVT_HAARFEATURE         *pInternalFeature;
	int32_t                     nSum, nArea;
	int32_t                     nCorrectionRatio;
	int32_t                     i;
	ADI_RECTANGLE               oRect[3];
	int32_t                     nRectangles;
	int32_t                     nResize;

	nSum = 0;
	nArea = 0;

	pIntegralImage = pImageData->pIntegralImage;
	pSquareIntegralImage = pImageData->pSquareIntegral;
	pInternalClassifierCascade = pCascadeClassifier->pPvtCascade;

	pCascadeClassifier->nScale = nScale;
	pCascadeClassifier->oRealWindowSize.nWidth = adi_mult1616_32bit_o16(nScale,
		pCascadeClassifier->oOriginalWindowSize.nWidth);

	pCascadeClassifier->oRealWindowSize.nHeight = adi_mult1616_32bit_o16(nScale,
		pCascadeClassifier->oOriginalWindowSize.nHeight);

	pInternalClassifierCascade->pSum = pImageData->pIntegralImage;
	pInternalClassifierCascade->pSqSum = pImageData->pSquareIntegral;

	nResize = nScale & 0xFFFF;
	if (nResize > 0x7FFF)
	{
		nPointX = 1;
		nPointY = 1;
	}
	else
	{
		nPointX = 0;
		nPointY = 0;
	}

	nResize = nScale >> 16;
	nPointX += nResize;
	nPointY += nResize;
	nRectWidth = adi_mult1616_32bit_o16(nScale,
		(pCascadeClassifier->oOriginalWindowSize.nWidth - 2));
	nRectHeight = adi_mult1616_32bit_o16(nScale,
		(pCascadeClassifier->oOriginalWindowSize.nHeight - 2));
	nWeightScale = ADI_MAX_VALUE_131 / (nRectWidth * nRectHeight);  /* In 1.31 Format */

	pInternalClassifierCascade->nInverseWindowArea = nWeightScale;

	pInternalClassifierCascade->p0 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * nPointY) + nPointX);
	pInternalClassifierCascade->p1 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * nPointY) +(nPointX + nRectWidth));
	pInternalClassifierCascade->p2 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + nPointX);
	pInternalClassifierCascade->p3 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + (nPointX + nRectWidth));

	pInternalClassifierCascade->pq0 = (uint64_t *) (pSquareIntegralImage + (pImageData->oImageSize.nWidth * nPointY) + nPointX);
	pInternalClassifierCascade->pq1 = (uint64_t *) (pSquareIntegralImage + (pImageData->oImageSize.nWidth * nPointY) + (nPointX + nRectWidth));
	pInternalClassifierCascade->pq2 = (uint64_t *) (pSquareIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + nPointX);
	pInternalClassifierCascade->pq3 = (uint64_t *) (pSquareIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + (nPointX + nRectWidth));

	/* Initialize pointers in haar features according to real window size and
	given image pointers */
	for (i = 0; i < pCascadeClassifier->nCount; i++)
	{
		for
			(
			nLoop1 = 0;
		nLoop1 < pInternalClassifierCascade->pStageClassifier[i].nCount;
		nLoop1++
			)
		{
			for
				(
				nLoop3 = 0;
			nLoop3 < pInternalClassifierCascade->pStageClassifier[i].pClassifier[nLoop1].nCount;
			nLoop3++
				)
			{
				pFeature = &pCascadeClassifier->pStageClassifier[i].pClassifier[nLoop1].pHaarFeature[nLoop3];
				pInternalFeature = &pInternalClassifierCascade->pStageClassifier[i].pClassifier[nLoop1].pNode[nLoop3].oFeature;
				nSum = 0;
				nArea = 0;

				/* align blocks */
				for
					(
					nLoop2 = 0;
				nLoop2 < ADI_MAX_HAAR_RECTANGLE_FEATURES;
				nLoop2++
					)
				{
					if (pInternalFeature->oPvtHaarRect[nLoop2].p0 == NULL)
					{
						break;
					}

					oRect[nLoop2] = pFeature->nHaarRectangle[nLoop2].nRect;
				}

				nRectangles = nLoop2;
				for (nLoop2 = 0; nLoop2 < nRectangles; nLoop2++)
				{
					nPointX = adi_mult1616_32bit_o16(nScale,
						oRect[nLoop2].nX);
					nRectWidth = adi_mult1616_32bit_o16(nScale,
						oRect[nLoop2].nWidth);
					nPointY = adi_mult1616_32bit_o16(nScale,
						oRect[nLoop2].nY);
					nRectHeight = adi_mult1616_32bit_o16(nScale,
						oRect[nLoop2].nHeight);
					nCorrectionRatio = nWeightScale;
					pInternalFeature->oPvtHaarRect[nLoop2].p0 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * nPointY) + nPointX);
					pInternalFeature->oPvtHaarRect[nLoop2].p1 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * nPointY) + (nPointX + nRectWidth));
					pInternalFeature->oPvtHaarRect[nLoop2].p2 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + nPointX);
					pInternalFeature->oPvtHaarRect[nLoop2].p3 = (uint32_t *) (pIntegralImage + (pImageData->oImageSize.nWidth * (nPointY + nRectHeight)) + (nPointX + nRectWidth));
					pInternalFeature->oPvtHaarRect[nLoop2].nWeight = adi_mult1616_131(pFeature->nHaarRectangle[nLoop2].nWeight,
						nCorrectionRatio);
					if (nLoop2 == 0)
					{
						nArea = ADI_MAX_VALUE_131 / (nRectWidth * nRectHeight);
					}
					else
					{
						nSum += adi_mult1616_32bit(pInternalFeature->oPvtHaarRect[nLoop2].nWeight,
							nRectWidth * nRectHeight);
					}
				}

				pInternalFeature->oPvtHaarRect[0].nWeight = adi_mult1616_131(nSum,
					nArea);
				pInternalFeature->oPvtHaarRect[0].nWeight = adi_mult1616_32bit(pInternalFeature->oPvtHaarRect[0].nWeight,
					-1);
			}
		}
	}
}

int32_t adi_EvaluateSubWindow(
	const ADI_HAARCLASSIFIERCASCADE *pCascadeClassifier,
	const ADI_IMAGE_HAAR_DATA       *pImageData,
	ADI_POINT                       oStartPoint,
	int32_t                         nStartStage
	)
{
	int32_t                     nReturnResult;
	int32_t                     nSumOffset, nSqSumOffset;
	int32_t                     i, j;
	int32_t                     nMean, nVariance;
	int32_t                     nStageSum;
	//int32_t                     nMeanSquare;
	ADI_PVT_CLASSIFIERCASCADE   *pPvtClassifierCascade;
	ADI_PVT_HAARCLASSIFIER      *pPvtClassifier;
	ADI_PVT_HAARTREENODE        *pPvtNode;
	int32_t                     nAccumSum, nThreshold, nLeftWeight, nRightWeight;
	int32_t                     nSum, nDifference, nResult;
	int32_t                     nIntegralSum;

	nReturnResult = 1;
	pPvtClassifierCascade = pCascadeClassifier->pPvtCascade;
	nSumOffset = (oStartPoint.nY * pImageData->oImageSize.nWidth) + oStartPoint.nX;
	nSqSumOffset = (oStartPoint.nY * pImageData->oImageSize.nWidth) + oStartPoint.nX;
	nIntegralSum = (pPvtClassifierCascade->p0[nSumOffset]
	- (pPvtClassifierCascade->p1[nSumOffset] + pPvtClassifierCascade->p2[nSumOffset]))
		+ pPvtClassifierCascade->p3[nSumOffset];

	nMean = adi_mult131_32bit(nIntegralSum,
		pPvtClassifierCascade->nInverseWindowArea);
	nVariance = ((pPvtClassifierCascade->pq0[nSqSumOffset]
	- (pPvtClassifierCascade->pq1[nSqSumOffset] + pPvtClassifierCascade->pq2[nSqSumOffset]))
		+ pPvtClassifierCascade->pq3[nSqSumOffset]);

	nVariance = adi_mult131_32bit(nVariance,
		pPvtClassifierCascade->nInverseWindowArea);
	nVariance = adi_SqrRootFixed(nVariance,
		16);
	nSum = nVariance + nMean;
	nDifference = nVariance - nMean;
	nResult = adi_mult1616_1616(nSum,
		nDifference);
	if (nResult > 0)
	{
		nVariance = adi_SqrRootFixed(nResult,
			16);
	}
	else
	{
		nVariance = ADI_ONE_IN1616;
	}

	for (i = nStartStage; i < pPvtClassifierCascade->nCount; i++)
	{
		nStageSum = 0;
		for (j = 0; j < pPvtClassifierCascade->pStageClassifier[i].nCount; j++)
		{
			pPvtClassifier = pPvtClassifierCascade->pStageClassifier[i].pClassifier + j;
			pPvtNode = pPvtClassifier->pNode;
			nThreshold = adi_mult1616_1616(pPvtNode->nThreshold,
				nVariance);
			nIntegralSum = (pPvtNode->oFeature.oPvtHaarRect[0].p0[nSumOffset]
			- (pPvtNode->oFeature.oPvtHaarRect[0].p1[nSumOffset] + pPvtNode->oFeature.oPvtHaarRect[0].p2[nSumOffset]))
				+ pPvtNode->oFeature.oPvtHaarRect[0].p3[nSumOffset];

			nAccumSum = adi_mult1616_32bit(nIntegralSum,
				pPvtNode->oFeature.oPvtHaarRect[0].nWeight);
			nIntegralSum = (pPvtNode->oFeature.oPvtHaarRect[1].p0[nSumOffset]
			- (pPvtNode->oFeature.oPvtHaarRect[1].p1[nSumOffset] + pPvtNode->oFeature.oPvtHaarRect[1].p2[nSumOffset]))
				+ pPvtNode->oFeature.oPvtHaarRect[1].p3[nSumOffset];

			nAccumSum += adi_mult1616_32bit(nIntegralSum,
				pPvtNode->oFeature.oPvtHaarRect[1].nWeight);
			if (pPvtNode->oFeature.oPvtHaarRect[2].p0 != NULL)
			{
				nIntegralSum = (pPvtNode->oFeature.oPvtHaarRect[2].p0[nSumOffset]
				- (pPvtNode->oFeature.oPvtHaarRect[2].p1[nSumOffset] + pPvtNode->oFeature.oPvtHaarRect[2].p2[nSumOffset]))
					+ pPvtNode->oFeature.oPvtHaarRect[2].p3[nSumOffset];

				nAccumSum += adi_mult1616_32bit(nIntegralSum,
					pPvtNode->oFeature.oPvtHaarRect[2].nWeight);
			}

			nLeftWeight = pPvtClassifier->pAlpha[0];
			nRightWeight = pPvtClassifier->pAlpha[1];
			nStageSum += (nAccumSum < nThreshold) ? (nLeftWeight) : (nRightWeight);
		}

		if (nStageSum < pPvtClassifierCascade->pStageClassifier[i].nThreshold)
		{
			nReturnResult = -i;
			return (nReturnResult);
		}
	}

	return (nReturnResult);
}

int32_t adi_mult1616_32bit_o16(
	int32_t nNum1,
	int32_t nNum2
	)
{
	int32_t nResult;
	nResult = (((int64_t) nNum1 * nNum2));
	nResult += 0x8000;
	nResult >>= 16;
	return (nResult);
}

int32_t adi_mult1616_131(
	int32_t nNum1,
	int32_t nNum2
	)
{
	int32_t nResult;
	nResult = (((int64_t) nNum1 * nNum2) >> 31);
	return (nResult);
}

int32_t adi_mult1616_32bit(
	int32_t nNum1,
	int32_t nNum2
	)
{
	int32_t nResult;
	nResult = (((int64_t) nNum1 * nNum2));
	return (nResult);
}

int32_t adi_mult131_32bit(
	int32_t nNum1,
	int32_t nNum2
	)
{
	int32_t nResult;
	nResult = ((int64_t) nNum1 * nNum2) >> 15;
	return (nResult);
}

int32_t adi_mult1616_1616(
	int32_t nNum1,
	int32_t nNum2
	)
{
	int32_t nResult;
	nResult = ((int64_t) nNum1 * nNum2) >> 16;
	return (nResult);
}

uint32_t adi_SqrRootFixed(
	uint32_t    nNum,
	uint32_t    nShift
	)
{
	int16_t     nSignBits;
	int16_t     nFractPos;
	int16_t     nTemp1;
	int16_t     nFract16;

	uint32_t    nFract;

	/* Calculate the number of sign bits */
	nSignBits = norm_fr1x32(nNum);

	nFractPos = 32 - nShift;

	if (nFractPos <= nSignBits)
	{

		/* Number is a fraction so normalize and pass to sqrt_fr16*/
		nFract = nNum << nFractPos;
		nFract = nFract >> 17;

		nFract16 = nFract;
		nFract16 = _sqrt_fr16(nFract16);
		nFract = nFract16;

		nFract = nFract << 17;
		nFract = nFract >> nFractPos;
	}
	else
	{
		nFract = nNum << nSignBits;
		nTemp1 = nFractPos - nSignBits;

		if ((nTemp1 & 0x0001) > 0)
		{
			nFract = nFract >> 1;
			nTemp1 = nTemp1 + 1;
		}

		nTemp1 = nTemp1 >> 1;
		nFractPos = nFractPos - nTemp1;
		nFract = nFract >> 17;

		nFract16 = nFract;
		nFract16 = _sqrt_fr16(nFract16);
		nFract = nFract16;

		nFract = nFract << 17;
		nFract = nFract >> nFractPos;
	}

	return (nFract);
}

int16_t _sqrt_fr16(
	int16_t  nInp
	)
{
	int16_t *pCoeff;
	int16_t nNormInp;
	int16_t nShifts, nShiftsOut, nOutput;
	int32_t nResult, nTemp1, nTemp2;
	int32_t nIPrecision;

	/* Precision Required (High Value = Low Precision and vice versa */
	nIPrecision = 0;

	/* Check for 0 and negative conditions and return 0 if true */
	if (nInp <= 0)
	{
		return (0);
	}

	/* Find the Shifts */
	nShifts = norm_fr1x16(nInp);
	nShiftsOut = nShifts >> 1;
	nShifts = nShiftsOut << 1;

	/* Normalize the Input */
	nNormInp = nInp << nShifts;

	/* Select Coefficient based on the Number */
	nNormInp = nNormInp - 0x4000;
	if (nNormInp < 0)
	{
		pCoeff = sqrtcoef0;
		nNormInp = -1 * nNormInp;
	}
	else
	{
		pCoeff = sqrtcoef1;
	}

	/* nResult = xs^0*Coeff0 */
	nResult = pCoeff[0] << (16 - nIPrecision);

	/* nResult = xs^0*Coeff0 + xs^1*Coeff1 */
	nTemp2 = ASM_MUL16(nNormInp, pCoeff[1]);
	nTemp2 = nTemp2 >> nIPrecision;
	nResult = nResult + nTemp2;

	/* nResult = xs^0*Coeff0 + xs^1*Coeff1 + xs^2*Coeff2 */
	nTemp1 = ASM_MUL16(nNormInp, nNormInp);
	nTemp1 = UBROUND2POS(nTemp1, 16);
	nTemp1 = nTemp1 >> 16;

	nTemp2 = ASM_MUL16(nTemp1, pCoeff[2]);
	nTemp2 = nTemp2 >> nIPrecision;
	nResult = nResult + nTemp2;

	/* nResult = xs^0*Coeff0 + xs^1*Coeff1 + xs^2*Coeff2 + xs^3*Coeff3 */
	nTemp1 = ASM_MUL16(nTemp1, nNormInp);
	nTemp1 = UBROUND2POS(nTemp1, 16);
	nTemp1 = nTemp1 >> 16;

	nTemp2 = ASM_MUL16(nTemp1, pCoeff[3]);
	nTemp2 = nTemp2 >> nIPrecision;
	nResult = nResult + nTemp2;

	/* nResult = xs^0*Coeff0 + xs^1*Coeff1 + xs^2*Coeff2 + xs^3*Coeff3 + xs^4*Coeff4 */
	nTemp1 = ASM_MUL16(nTemp1, nNormInp);
	nTemp1 = UBROUND2POS(nTemp1, 16);
	nTemp1 = nTemp1 >> 16;

	nTemp2 = ASM_MUL16(nTemp1, pCoeff[4]);
	nTemp2 = nTemp2 >> nIPrecision;
	nResult = nResult + nTemp2;

	if (nIPrecision > 15)
	{
		nResult = nResult << (nIPrecision - 15);
	}
	else
	{
		nResult = nResult >> (15 - nIPrecision);
	}

	nOutput = nResult >> nShiftsOut;

	return (nOutput);
}
