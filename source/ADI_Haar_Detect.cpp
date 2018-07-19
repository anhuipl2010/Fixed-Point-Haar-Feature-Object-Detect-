// ADI_Haar_Detect.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "adi_image_tool_box.h"
#include "adi_haarfeatures.h"
//#include "adi_util_funcs.h"
#include "profile.h"

#include <opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <iostream> 

/*=============  D A T A  =============*/
ALIGN(4)
	MEMORY_SECTION(adi_slow_noprio_disp_rw)
	int8_t  aInputImage[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT * 3];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	uint16_t    aImageSquare[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint8_t aGrayImage[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint32_t    aIntegralImage[ADI_ROI_IMAGE_WIDTH * ADI_ROI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint64_t    aIntegralImageSquare[ADI_ROI_IMAGE_WIDTH * ADI_ROI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(vidout_section0)
	uint32_t    aImageTemp[ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	uint16_t    aDetectedFaces[ADI_MAX_FACES * 4];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	char_t  aTrainedDataMemory[ADI_MEMORYFOR_TRAINEDDATA];

ALIGN(4)
	MEMORY_SECTION(adi_slowb1_prio0_rw)
	int32_t aTrainedFileMemory[ADI_TRAINED_FILE_SIZE];

ALIGN(4)
	MEMORY_SECTION(adi_slow_noprio_disp_rw)
	char_t          aString[PPM_HEADER_SIZE];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   aOutFilename[256];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   atempFilename[256];

ALIGN(4)
	MEMORY_SECTION(adi_appl_slow_noprio_rw)
	static char_t   aPathToMedia[256];
/******************************************************************************

  Function              : main

  Function description  : Setup path for media files,
                          Check Input file type,
                          Open Input file and read data into L3 buffer,
                          Call required module,
                          Open Output file and write processed data


  Parameters            : None


  Returns               : Status (Zero/Non-Zero)

  Notes                 :

******************************************************************************/

void test_haar_default()
{
	char input_image_name[100];
	char output_image_name[100];
	int result_num = 0;
	char result_name[100];
	int image_num = NUM_INPUT_FILES;
	cv::CascadeClassifier classifier;
	classifier.load("cascade0716_faec_24x12-16.xml");
	std::vector<cv::Rect> results;

	for (int i = 0; i <= image_num; i++)
	{
		sprintf(input_image_name,"../%s\\%d.jpg","image",i); 
		cv::Mat input_image = cv::imread(input_image_name);

		if (input_image.empty())
		{
			continue;
		}

		classifier.detectMultiScale(input_image,results,1.1,2,CV_HAAR_SCALE_IMAGE,cv::Size(24,12)/*,cv::Size(15,15)*/);

		for (int j = 0; j < results.size(); j++)
		{
			rectangle(input_image,results[j],cv::Scalar(0,255,0),2);
		}
		results.clear();

		sprintf(output_image_name,"../%s\\%d.bmp","OpenCV_result",i); 
		//imwrite(output_image_name,input_image);
		cv::imshow("opencv--result",input_image);
		cv::waitKey(1);
	}
}

extern int32_t haar_features_params[];

int _tmain(int argc, _TCHAR* argv[])
{
	test_haar_default();

	FILE                        *pInputFile;
	char_t                      aType[32];
	char_t                      aExtension[8];
	int32_t                     nResult;
	uint32_t                    nFacesDetecetd;
	ADI_IMAGE_HAAR_DATA         oImageData;
	int32_t                     i, j;
	float32_t                   nScaleIncrement;
	ADI_IMAGE_SIZE              oMinObjectSize;
	uint32_t                    nBytesRead;
	uint32_t                    nDisplayColor;
	ADI_POINT                   nCenter;
	int32_t                     nRadius;
	uint8_t                     *pImagePtr;
	uint32_t                    nXOffset;
	uint32_t                    nYOffset;
	uint32_t                    nStride;
	uint32_t                    nNumRawFaces;
	uint32_t                    nMinNeighbours;
	uint32_t                    nOvelapPercentage;
	FILE                        *pOutFile;
	ADI_HAARCLASSIFIERCASCADE   *pClassifierCascade;


	char input_image_name[100];
	char output_image_name[100];

	memcpy(aTrainedFileMemory,haar_features_params,ADI_TRAINED_FILE_SIZE * sizeof(int32_t));

	nResult = adi_HaarFeaturesInit((ADI_HAARCLASSIFIERCASCADE *)aTrainedDataMemory,
		ADI_MEMORYFOR_TRAINEDDATA,
		(int8_t *)aTrainedFileMemory,
		ADI_TRAINED_FILE_SIZE);

	if (nResult < 0)
	{
		printf("Memory Given For Storing Trained data was not sufficient. %d bytes short", -(nResult));
		return -1;
	}

	for (i = 1; i < NUM_INPUT_FILES; i++)
	{
		//Load input image
		sprintf(input_image_name,"../%s\\%d.jpg","image",i); 
		cv::Mat input_image = cv::imread(input_image_name);

		if (input_image.empty())
		{
			break;
		}
		memcpy(aInputImage, input_image.data, ADI_IMAGE_WIDTH * ADI_IMAGE_HEIGHT * 3 * sizeof(uint8_t));

	
		printf("Image Num = %d, process => Haar Feature Face Detection\n", i);

		nScaleIncrement = 1.4567;
		nXOffset = 0;
		nYOffset = 0;
		nMinNeighbours = 0;
		nOvelapPercentage = 75;
		
		//memcpy(aTrainedFileMemory,haar_features_params,ADI_TRAINED_FILE_SIZE * sizeof(int32_t));

		adi_RGB2GRAY((uint8_t *)aInputImage,
			aGrayImage,
			ADI_IMAGE_WIDTH ,
			ADI_IMAGE_HEIGHT);

		pImagePtr = aGrayImage;
		pImagePtr = pImagePtr + (nYOffset * ADI_IMAGE_WIDTH) + nXOffset;
		nStride = ADI_IMAGE_WIDTH - ADI_ROI_IMAGE_WIDTH;

		adi_HaarPreProcess(&oImageData,
			pImagePtr,
			aImageSquare,
			aIntegralImage,
			aIntegralImageSquare,
			aImageTemp,
			ADI_ROI_IMAGE_WIDTH,
			ADI_ROI_IMAGE_HEIGHT,
			nStride);

		/*nResult = adi_HaarFeaturesInit((ADI_HAARCLASSIFIERCASCADE *)aTrainedDataMemory,
			ADI_MEMORYFOR_TRAINEDDATA,
			(int8_t *)aTrainedFileMemory,
			ADI_TRAINED_FILE_SIZE);

		if (nResult < 0)
		{
			printf("Memory Given For Storing Trained data was not sufficient. %d bytes short", -(nResult));
			return -1;
		}*/

		pClassifierCascade = (ADI_HAARCLASSIFIERCASCADE *)aTrainedDataMemory;

		/* Optionally You can add adi_EqualizeHist() Here to improve the results */
		oMinObjectSize.nWidth = ADI_MIN_OBJECTWIDTH;
		oMinObjectSize.nHeight = ADI_MIN_OBJECTHEIGHT;

		nNumRawFaces = adi_HaarDetectObjects(&oImageData,
			nScaleIncrement,
			pClassifierCascade,
			&oMinObjectSize,
			aDetectedFaces,
			(uint8_t *)aImageTemp);

		nFacesDetecetd = adi_HaarPostProcess(aDetectedFaces,
			(uint16_t *)aImageTemp,
			nNumRawFaces,
			nOvelapPercentage,
			nMinNeighbours);

		//Draw the results
		//printf("XCoOrdinate   YCoOrdinate  Radius \n");
		for (j = 0; j < nFacesDetecetd * 4; j += 4)
		{
			//nCenter.nX = adi_Round((float32_t) (aDetectedFaces[j] + nXOffset + aDetectedFaces[j + 2] * 0.5));
			//nCenter.nY = adi_Round((float32_t) (aDetectedFaces[j + 1] + nYOffset + aDetectedFaces[j + 3] * 0.5));
			//nRadius = adi_Round((aDetectedFaces[j + 2] + aDetectedFaces[j + 3]) * 0.25);

			//printf("%4d         %4d         %4d\n",nCenter.nX,nCenter.nY,nRadius);
			//if (nRadius >= ADI_MIN_OBJECTWIDTH)
			//{
				//circle(input_image,cv::Point(nCenter.nX, nCenter.nY), nRadius, cv::Scalar(0,255,0),2);
			//}

			cv::Rect result_rect;
			result_rect.x = (int)(aDetectedFaces[j]);
			result_rect.y = (int)(aDetectedFaces[j + 1]);
			result_rect.width = (int)(aDetectedFaces[j + 2]);
			result_rect.height = (int)(aDetectedFaces[j + 3]);

			//if (result_rect.width > ADI_MIN_OBJECTWIDTH * 2)
			{
				cv::rectangle(input_image, result_rect, cv::Scalar(0, 255, 0), 2, 8, 0);
			}
			

		}
		sprintf(output_image_name,"../%s\\%d.jpg","ADI_result",i); 
		//imwrite(output_image_name,input_image);
		cv::imshow("adi--result",input_image);
		cv::waitKey(1000);
	}
	
	printf("*******************************************************************\n");
	printf("Done\n");
	return 0;
}

