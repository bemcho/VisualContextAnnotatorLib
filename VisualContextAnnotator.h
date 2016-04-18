#pragma once
#include<string>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <thread>

#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.h"
using namespace  std;
using namespace cv;

using namespace std;

class VisualContextAnnotator
{
public:
	VisualContextAnnotator();
	virtual ~VisualContextAnnotator();
	void loadCascadeClassifier(const string cascadeClassifierPath);
	void loadLBPModel(const string path);
	void loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath);
	void detectWithCascadeClassifier(vector<Rect>& result, Mat& frame_gray);
	Annotation predictWithLBPInRectangle(const Rect & detect, Mat & frame_gray);
	void predictWithLBP(vector<Annotation>& annotations, cv::Mat & frame_gray);
	void predictWithCAFFE(Annotation & annotation, cv::Mat & frame);

private:
	CascadeClassifier cascade_classifier;
	Ptr<face::FaceRecognizer> model;
	dnn::Net net;
	void getMaxClass(dnn::Blob & probBlob, int * classId, double * classProb);
	std::vector<String> readClassNames(const string filename);
	std::vector<String> classNames;
};

