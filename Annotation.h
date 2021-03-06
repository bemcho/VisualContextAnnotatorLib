#pragma once
#include<string>
#include "opencv2/core.hpp"
class Annotation
{
public:
	Annotation();
	Annotation(cv::Rect rect, std::string desc);
	virtual ~Annotation();
	std::string getDescription();
	cv::Rect getRectangle();
	void setDescription(std::string desc);
	void setRectangle(cv::Rect rect);
private:
	cv::Rect rectangle;
	std::string description;
};

