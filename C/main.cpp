#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include <fstream>
#include <ctime>
using namespace std;
using namespace cv;
int main(void) {

	int cntFrame = 0;
	string NIR_Path = "";
	string RGB_Path = "";
	string NDVI_pathToSave = "";
	string NDWI_pathToSave = "";
	string pathTosave_NDVI = "";
	int i, j, k, width, height;
	Mat src1, src2, R_band, G_band, NIR_band, bgr1[3], nir_b[3], m1, m2;

	clock_t t1, t2, t3, t4;
	double temps;
	long clk_tck = CLOCKS_PER_SEC;
	while (1) {

		printf(" %d :\n", cntFrame);
		NIR_Path = "NIR/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		RGB_Path = "RGB/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		NDVI_pathToSave = "NDVI/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		NDWI_pathToSave = "NDWI/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";



		cntFrame++;
	
		// BF1: Acquisition et separation des Bondes
		//******************************************************************************************************************
		t1 = clock();
		src1 = imread(RGB_Path, CV_LOAD_IMAGE_COLOR);

		if (src1.empty()) { waitKey(); break; }
		
		src2 = imread(NIR_Path, CV_LOAD_IMAGE_COLOR);
		if (src2.empty()) { waitKey(); break; }

		Size size(512, 512);

		resize(src1, src1, size);
		resize(src2, src2, size);

		split(src1, bgr1);
		split(src2, nir_b);

		 G_band = bgr1[1];
		 R_band = bgr1[2];
		 NIR_band = nir_b[0];
		 

	
		 width = src1.cols;
		 height = src1.rows;
		
		float *R = (float*)malloc(sizeof(float)*width *height);
		float *G = (float*)malloc(sizeof(float)*width *height);
		float *NIR = (float*)malloc(sizeof(float)*width *height);
		float *NDVI = (float*)malloc(sizeof(float)*width *height);
		float *NDWI = (float*)malloc(sizeof(float)*width *height);

	
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				NIR[width*i + j] = (float)NIR_band.at<uchar>(i, j)/ 255;
				R[width*i + j] = (float)R_band.at<uchar>(i, j) / 255;
				G[width*i + j] = (float)G_band.at<uchar>(i, j) / 255;
			}
		}

		t2 = clock();
		temps = (double)(t2 - t1) / clk_tck;
		printf("temps d'exécution de bloc 1 = %f\n", temps);

		// BF2: Calcul des index NDVI NDWI 
		//******************************************************************************************************************
		t1 = clock();
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				NDVI[width*i + j] = (NIR[width*i + j] - R[width*i + j]) / (NIR[width*i + j] + R[width*i + j]);
				NDWI[width*i + j] = (G[width*i + j] - NIR[width*i + j]) / (G[width*i + j] + NIR[width*i + j]);

			}
		}
		t2 = clock();
		temps = (double)(t2 - t1) / clk_tck;
		printf("temps d'exécution de bloc 2 = %f\n", temps);
		// FB3 : Opération de seuillage et stockage des images.
		//******************************************************************************************************************
		t1 = clock();
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {

				if (NDVI[width*i + j] < 0.6)  NDVI[width*i + j] = 0;
				else NDVI[width*i + j] = 255;

				if (NDWI[width*i + j] < 0.2) NDWI[width*i + j] = 0;
				else  NDWI[width*i + j] = 255;
			}
		}


		vector <uchar>vec1(width * height);
		vector <uchar> vec2(width * height);
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				vec1[width*i + j] = NDVI[width*i + j];
				vec2[width*i + j] = NDWI[width*i + j];


			}
		}
		
		m1 = Mat(height, width, CV_8UC1);
		m2 = Mat(height, width, CV_8UC1);
		memcpy(m1.data, vec1.data(), vec1.size() * sizeof(uchar));
		memcpy(m2.data, vec2.data(), vec2.size() * sizeof(uchar));


		cv::imwrite(NDVI_pathToSave, m1);
		cv::imwrite(NDWI_pathToSave, m2);
		t2 = clock();
		temps = (double)(t2 - t1) / clk_tck;
		printf("temps d'exécution de bloc 3 = %f\n", temps);
		
		imshow("NDVI Image", m1);
		//imshow("NDWI Image", m2);
		waitKey(0);
		
		free(R);
		free(G);
		free(NIR);
		free(NDVI);
		free(NDWI);
		


	}
	


	return 0;
}