#include "VisualContextAnnotator.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
VisualContextAnnotator::VisualContextAnnotator()
{
	model = face::createLBPHFaceRecognizer();
}


VisualContextAnnotator::~VisualContextAnnotator()
{
	cascade_classifier.~CascadeClassifier();
	model.release();
	net.~Net();
}

void VisualContextAnnotator::loadCascadeClassifier(const string cascadeClassifierPath)
{
	//-- 1. Load the cascade
	if (!cascade_classifier.load(cascadeClassifierPath) || cascade_classifier.empty()) { printf("--(!)Error loading face cascade\n"); };

}

void VisualContextAnnotator::loadLBPModel(const string path)
{
	model->load(path);
}

void VisualContextAnnotator::loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath)
{
	Ptr<dnn::Importer> importer;
	try                                     //Try to import Caffe GoogleNet model
	{
		importer = dnn::createCaffeImporter(modelProtoTextPath, modelBinPath);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}
	if (!importer)
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelProtoTextPath << std::endl;
		std::cerr << "caffemodel: " << modelBinPath << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
	}

	importer->populateNet(net);

	importer.release();
	classNames = readClassNames(synthWordPath);
}
void VisualContextAnnotator::detectWithCascadeClassifier(vector<Rect>& result, Mat & frame_gray)
{
	cascade_classifier.detectMultiScale(frame_gray, result, 1.1, 3, 0, Size(50, 50), Size());
}
void predictInThreadCAFFE(VisualContextAnnotator& annotator, Annotation& annotation, Mat& frame)
{
	annotator.predictWithCAFFE(annotation, frame);
}

void predictInThreadLBP(Ptr<face::FaceRecognizer> model, Annotation& result, const Rect& detect, Mat& frame_gray)
{

}
Annotation VisualContextAnnotator::predictWithLBPInRectangle(const Rect& detect, Mat& frame_gray)
{
	Mat face = frame_gray(detect);
	int predictedLabel = -1;
	double confidence = 0.0;
	model->predict(face, predictedLabel, confidence);
	std::stringstream fmt;
	if (predictedLabel > 0)
	{
		fmt << model->getLabelInfo(predictedLabel) << "L:" << predictedLabel << "C:" << confidence;
	}
	else
	{
		fmt << "Unknown Human" << "L:" << predictedLabel << "C:" << confidence;
	}
	return Annotation(detect, fmt.str());
}

struct PredictWithLBPBody {
	VisualContextAnnotator & vca_;
	vector<Rect> detects_;
	Mat& frame_gray_;
	Annotation* result_;
	PredictWithLBPBody(VisualContextAnnotator & u, vector<Rect> detects, Mat& frame_gray) :vca_(u), detects_(detects), frame_gray_(frame_gray) {}
	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); ++i)
			result_[i] = vca_.predictWithLBPInRectangle(detects_[i], frame_gray_);
	}
};

void VisualContextAnnotator::predictWithLBP(vector<Annotation>& annotations, cv::Mat& frame_gray)
{
	static tbb::affinity_partitioner affinity;

	vector<Rect> detects;
	detectWithCascadeClassifier(detects, frame_gray);
	PredictWithLBPBody parallelLBP(*this, detects, frame_gray);

	const int tsize = detects.size();

	parallelLBP.result_ = new Annotation[tsize];
	vector<Annotation> result(tsize);
	tbb::parallel_for(tbb::blocked_range<int>(0, tsize), // Index space for loop
		parallelLBP,                    // Body of loop
		affinity);

	annotations = vector<Annotation>(parallelLBP.result_, parallelLBP.result_ + tsize);
}

void VisualContextAnnotator::predictWithCAFFE(Annotation& annotation, cv::Mat & frame)
{
	Mat img;
	resize(frame, img, Size(244, 244));
	dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch
	net.setBlob(".data", inputBlob);        //set the network input
	net.forward();                          //compute output
	dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer
	int classId;
	double classProb;
	getMaxClass(prob, &classId, &classProb);//find the best class

	// Calculate the position for annotated text (make sure we don't
	// put illegal values in there):
	stringstream caffe_fmt = stringstream();
	caffe_fmt << "Probability: " << classProb * 100 << "%" << std::endl;
	caffe_fmt << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
	annotation.setDescription(caffe_fmt.str());
}

/* Find best class for the blob (i. e. class with maximal probability) */
void VisualContextAnnotator::getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}
std::vector<String> VisualContextAnnotator::readClassNames(const string filename = "synset_words.txt")
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}
