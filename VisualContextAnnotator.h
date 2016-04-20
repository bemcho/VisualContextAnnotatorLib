#pragma once
#include<string>
#include <vector>

#include <fstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/critical_section.h"
using namespace  std;
using namespace cv;

class VisualContextAnnotator
{
public:
	VisualContextAnnotator();
	virtual ~VisualContextAnnotator();
	void loadCascadeClassifier(const string cascadeClassifierPath);
	void loadLBPModel(const string path);
	void loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath);
	void detectWithCascadeClassifier(vector<Rect>& result, Mat& frame_gray, Size minSize = Size(150,150));
	void detectTextWithMorphologicalGradient(vector<Rect>& result, Mat& frame);
	void detectObjectsWithCanny(vector<Rect>& result, Mat& frame, double lowThreshold, Size minSize = Size(100, 100));
	Annotation predictWithLBPInRectangle(const Rect & detect, Mat & frame_gray);
	void predictWithLBP(vector<Annotation>& annotations, cv::Mat & frame_gray);
	void predictWithLBP(vector<Annotation>& annotations, vector<Rect> detects, cv::Mat & frame);
	void predictWithCAFFE(vector<Annotation>& annotations, cv::Mat & frame, cv::Mat & frame_gray);
	void predictWithCAFFE(vector<Annotation>& annotations, vector<Rect> detects, cv::Mat & frame);
	Annotation predictWithCAFFEInRectangle(const Rect & detect, Mat & frame);

private:
	CascadeClassifier cascade_classifier;
	Ptr<face::FaceRecognizer> model;
	dnn::Net net;

	void getMaxClass(dnn::Blob & probBlob, int * classId, double * classProb);
	std::vector<String> readClassNames(const string filename);
	std::vector<String> classNames;
};

