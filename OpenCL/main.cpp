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
const char *saxpy_kernel =
"__kernel void RGB(__global float* vector1, __global float* vector2,__global float* vector3,__global float* vector4,__global float* vector5){"
"int indx = get_global_id(0);"
"vector4[indx] = (vector1[indx] - vector2[indx]) / (vector1[indx] + vector2[indx]); "
"vector5[indx] = (vector3[indx] - vector1[indx]) / (vector3[indx] + vector1[indx]); "
"}";
int main(void) {

	int cntFrame = 0;
	string NIR_Path = "";
	string RGB_Path = "";
	string NDVI_pathToSave = "";
	string NDWI_pathToSave = "";
	string pathTosave_NDVI = "";

	int i, j, k, width, height;
	Mat src1, src2, R_band, G_band, NIR_band, bgr1[3], nir_b[3], m1, m2;

	double duration1, duration2, duration3, duration4;
	cl_platform_id * platforms = NULL;
	cl_uint     num_platforms;
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

	cl_device_id     *device_list = NULL;
	cl_uint           num_devices;

	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
	cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&saxpy_kernel, NULL, &clStatus);
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	cl_kernel kernel_01 = clCreateKernel(program, "RGB", &clStatus);
	while (1) {

		printf(" %d :\n", cntFrame);

		NIR_Path = "NIR/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		RGB_Path = "RGB/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		NDVI_pathToSave = "NDVI/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";
		NDWI_pathToSave = "NDWI/_2016-06-06-13-01-39_7_frame" + to_string(cntFrame) + ".png";


		cntFrame++;

		// BF1: Acquisition et separation des Bondes
		//******************************************************************************************************************
		src1 = imread(RGB_Path, CV_LOAD_IMAGE_COLOR);

		if (src1.empty()) { waitKey(); break; }
		
		src2 = imread(NIR_Path, CV_LOAD_IMAGE_COLOR);
		if (src2.empty()) { waitKey(); break; }

		Size size(512, 512);

		//resize(src1, src1, size);
		//resize(src2, src2, size);

		split(src1, bgr1);
		split(src2, nir_b);

		 G_band = bgr1[1];
		 R_band = bgr1[2];
		 NIR_band = nir_b[0];
		 


		 width = src1.cols;
		 height = src1.rows;
		 std::cout << "width  :" << width << "\n";
		 std::cout << "height  :" << height << "\n";

		float *R = (float*)malloc(sizeof(float)*width *height);
		float *G = (float*)malloc(sizeof(float)*width *height);
		float *NIR = (float*)malloc(sizeof(float)*width *height);
		float *NDVI = (float*)malloc(sizeof(float)*width *height);
		float *NDWI = (float*)malloc(sizeof(float)*width *height);
		float *NDVI_CL = (float*)malloc(sizeof(float)*width *height);
		float *NDWI_CL = (float*)malloc(sizeof(float)*width *height);

	
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				NIR[width*i + j] = (float)NIR_band.at<uchar>(i, j)/ 255;
				R[width*i + j] = (float)R_band.at<uchar>(i, j) / 255;
				G[width*i + j] = (float)G_band.at<uchar>(i, j) / 255;
			}
		}
		// BF2: Calcul des index NDVI NDWI 
		//******************************************************************************************************************
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				//NDVI[width*i + j] = (NIR[width*i + j] - R[width*i + j]) / (NIR[width*i + j] + R[width*i + j]);
				//NDWI[width*i + j] = (G[width*i + j] - NIR[width*i + j]) / (G[width*i + j] + NIR[width*i + j]);

			}
		}
		
		//1251 936//138//69//324//368//414//432//483
		size_t global_size = width*height;
		//size_t local_size = width*height;
		//size_t local_size = 69*2*2*2;
		size_t local_size = 504;

		cl_mem NIR_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height * sizeof(float), NULL, &clStatus);
		cl_mem R_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height * sizeof(float), NULL, &clStatus);
		cl_mem G_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height * sizeof(float), NULL, &clStatus);
		cl_mem NDVI_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *width*height, NULL, &clStatus);
		cl_mem NDWI_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *width*height, NULL, &clStatus);

		clStatus = clEnqueueWriteBuffer(command_queue, NIR_clmem, CL_TRUE, 0, width*height * sizeof(float), NIR, 0, NULL, NULL);
		clStatus = clEnqueueWriteBuffer(command_queue, R_clmem, CL_TRUE, 0, width*height * sizeof(float), R, 0, NULL, NULL);
		clStatus = clEnqueueWriteBuffer(command_queue, G_clmem, CL_TRUE, 0, width*height * sizeof(float), G, 0, NULL, NULL);

		duration1 = static_cast<double>(cv::getTickCount());

		clStatus = clSetKernelArg(kernel_01, 0, sizeof(cl_mem), (void *)&NIR_clmem);
		clStatus = clSetKernelArg(kernel_01, 1, sizeof(cl_mem), (void *)&R_clmem);
		clStatus = clSetKernelArg(kernel_01, 2, sizeof(cl_mem), (void *)&G_clmem);
		clStatus = clSetKernelArg(kernel_01, 3, sizeof(cl_mem), (void *)&NDVI_clmem);
		clStatus = clSetKernelArg(kernel_01, 4, sizeof(cl_mem), (void *)&NDWI_clmem);


		clStatus = clEnqueueNDRangeKernel(command_queue, kernel_01, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

		clStatus = clEnqueueReadBuffer(command_queue, NDVI_clmem, CL_TRUE, 0, width*height * sizeof(float), NDVI_CL, 0, NULL, NULL);
		clStatus = clEnqueueReadBuffer(command_queue, NDWI_clmem, CL_TRUE, 0, width*height * sizeof(float), NDWI_CL, 0, NULL, NULL);

		clStatus = clFlush(command_queue);
		clStatus = clFinish(command_queue);


		duration2 = static_cast<double>(cv::getTickCount()) - duration1;
		duration2 /= cv::getTickFrequency();

		std::cout << "\n temps de la regions parallel  :" << duration2 << "\n";
		// FB3 : Opération de seuillage et stockage des images.
		//******************************************************************************************************************
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {

				if (NDVI_CL[width*i + j] < 0.6)  NDVI_CL[width*i + j] = 0;
				else NDVI_CL[width*i + j] = 255;

				if (NDWI_CL[width*i + j] < -0.1) NDWI_CL[width*i + j] = 0;
				else  NDWI_CL[width*i + j] = 255;
			}
		}


		vector <uchar>vec1(width * height);
		vector <uchar> vec2(width * height);
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				vec1[width*i + j] = NDVI_CL[width*i + j];
				vec2[width*i + j] = NDWI_CL[width*i + j];


			}
		}

		m1 = Mat(height, width, CV_8UC1);
		m2 = Mat(height, width, CV_8UC1);
		memcpy(m1.data, vec1.data(), vec1.size() * sizeof(uchar));
		memcpy(m2.data, vec2.data(), vec2.size() * sizeof(uchar));

		imshow("NDVI Image", m1);
		imshow("NDWI Image", m2);
		waitKey(0);
		cv::imwrite(NDVI_pathToSave, m1);
		cv::imwrite(NDWI_pathToSave, m2);

		free(R);
		free(G);
		free(NIR);
		free(NDVI);
		free(NDWI);
		free(NDVI_CL);

		clStatus = clReleaseMemObject(NIR_clmem);
		clStatus = clReleaseMemObject(R_clmem);
		clStatus = clReleaseMemObject(NDVI_clmem);
		//clStatus = clReleaseMemObject(NDWI_clmem);
	}
	
	clStatus = clReleaseKernel(kernel_01);
	clStatus = clReleaseCommandQueue(command_queue);
	clStatus = clReleaseProgram(program);
	clStatus = clReleaseContext(context);

	return 0;
}