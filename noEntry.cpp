/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
// #include "hough.h"
#include "houghOOP.h"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame, const char *csvfilename, const char *imageName, string fileNum);
void parseCSV(const char *filename, vector<Rect> &actualSigns);
int countSuccess(std::vector<Rect> foundSigns, std::vector<Rect> actualSigns);
bool intersects(Rect a, Rect b);
bool majorityRed(Mat frame, float proportion);
bool minorityWhite(Mat frame, float proportion);

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;

/** Global constants */
const float UOI_THRESHOLD = 0.5;


/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image and Ground Truth
	std::string fileNumber = argv[1];
	std::string imageFile = "No_entry/NoEntry" + fileNumber + ".bmp";
	std::string groundTruthFile = "Sign_ground_truth/NoEntry" + fileNumber + ".csv";
	Mat frame = imread(imageFile, CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect No Entry Signs and Display Result
	detectAndDisplay(frame, groundTruthFile.c_str(), imageFile.c_str(), fileNumber);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );
	imwrite("DetectedSigns/detected" + fileNumber + ".jpg", frame);

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, const char *csvfilename, const char *imageName, string fileNum)
{
	std::vector<Rect> foundSigns;
	std::vector<Rect> actualSigns;
	parseCSV(csvfilename, actualSigns);
	Mat frame_gray;

	// 1. Prepare Image by converting to Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	std::vector<Rect> featureSigns;
	cascade.detectMultiScale( frame_gray, featureSigns, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

   // 3. Print number of Signs found
   printf("Number of found signs: %d \n", featureSigns.size());

   // 4. Select the signs detected by Viola-Jones
   // foundSigns = featureSigns;

	// 5. Perform circular Hough Transform Detection
	MyHoughCircles hc(100, 10, 110, 15);
	hc.performTransform(imageName, fileNum);
	std::vector<std::vector<int> > circles = hc.circles;

	// 6. Select feature signs that contain a circle centre
	for (int i=0; i < featureSigns.size(); i++) {
		bool valid = false;
		int j = 0;
		while (!valid && j < circles.size()) {
			Point centre = Point(circles[j][0], circles[j][1]);
			if (featureSigns[i].contains(centre)) {
				foundSigns.push_back(featureSigns[i]);
			}
			j++;
		}
	}

	// 7. Select feature signs that contain a circle centre and acceptable red-white colour proportion
	// for (int i=0; i < featureSigns.size(); i++) {
	// 	int j = 0;
	// 	while (j < circles.size()) {
	// 		Point centre = Point(circles[j][0], circles[j][1]);
	// 		if (featureSigns[i].contains(centre)) {
	// 			int x = featureSigns[i].x;
	// 			int y = featureSigns[i].y;
	// 			int w = featureSigns[i].width;
	// 			int h = featureSigns[i].height;
	// 			Range yRange = Range(max(y,0), min(y+h, frame.rows-1));
	// 			Range xRange = Range(max(x,0), min(x+w, frame.cols-1));
	// 			Mat subImage = frame(yRange, xRange);
	// 			if (majorityRed(subImage, 0.4) && minorityWhite(subImage, 0.6)) {
	// 				foundSigns.push_back(featureSigns[i]);
	// 				circles.erase(circles.begin() + j);
	// 				std::cout << "feature chosen" << std::endl;
	// 			}
	// 		}
	// 		j++;
	// 	}
	// }

	// 8. Select circle signs that contain a strong red-white proportion
	// for (int i=0; i < circles.size(); i++) {
	// 	int x = circles[i][0];
	// 	int y = circles[i][1];
	// 	int r = circles[i][2];
	// 	Range yRange = Range(max(y-r, 0), min(y+r, frame.rows-1));
	// 	Range xRange = Range(max(x-r, 0), min(x+r, frame.cols-1));
	// 	Mat subImage = frame(yRange, xRange);
	// 	if (majorityRed(subImage, 0.6) && minorityWhite(subImage, 0.4)) {
	// 		foundSigns.push_back(Rect(x-r, y-r, 2*r, 2*r));
	// 		std::cout << "circle chosen" << std::endl;
	// 	}
	// }

	// 9. Draw boxes around the actual signs
	for (int i=0; i < actualSigns.size(); i++) {
		rectangle(frame, Point(actualSigns[i].x, actualSigns[i].y), Point(actualSigns[i].x + actualSigns[i].width, actualSigns[i].y + actualSigns[i].height), Scalar( 0, 0, 255 ), 2);
	}

	// 10. Draw boxes around the found signs
	for (int i=0; i < foundSigns.size(); i++) {
		rectangle(frame, Point(foundSigns[i].x, foundSigns[i].y), Point(foundSigns[i].x + foundSigns[i].width, foundSigns[i].y + foundSigns[i].height), Scalar(0, 255, 0), 2);
	}
	printf("Number of filtered found signs: %d \n", foundSigns.size());

	// 11. Summarise the success rate
	float truePositives = countSuccess(foundSigns, actualSigns);
	float numTrue = actualSigns.size();
    float numPositive = foundSigns.size();
	float precision = truePositives / numPositive;
	float recall = truePositives / numTrue;
	float f1Score = 2 / ((1 / precision) + (1 / recall));

	std::cout << "TPR: " << recall << endl;
	std::cout << "F1 Score: " << f1Score << endl;
}

int countSuccess(std::vector<Rect> foundSigns, std::vector<Rect> actualSigns) {
	int numOfSuccess = 0;
	for (int i=0; i < foundSigns.size(); i++) {
		Rect foundSign = foundSigns[i];
		float foundArea = foundSign.area();
		for (int j=0; j < actualSigns.size(); j++) {
			Rect trueSign = actualSigns[j];
			float trueArea = trueSign.area();
			Rect intersection = foundSign & trueSign;
			float intersectionArea = intersection.area();
			if (intersection.area() != 0) {
				float uoi = intersectionArea / (foundArea + trueArea - intersectionArea);
				if (uoi > UOI_THRESHOLD) numOfSuccess++;
			}
		}
	}
	return numOfSuccess;
}

/** @function parseCSV */
void parseCSV(const char *filename, vector<Rect> &actualSigns) {
	std::fstream inStream;
	inStream.open(filename, ios::in);

	string line, value, temp;

	while (getline(inStream, line)) {

		string haystack = line;
		vector<int> values;
		size_t pos;
		while ((pos = haystack.find(',')) != string::npos) {
			int value = atoi(haystack.substr(0, pos).c_str());
			values.push_back(value);
			haystack.erase(0, pos + 1);
		}
		values.push_back(atoi(haystack.c_str()));
		assert(values.size() == 4);
		Rect actualSign = Rect(values[0], values[1], values[2], values[3]);
		actualSigns.push_back(actualSign);
	}
}

bool majorityRed(Mat frame, float threshold) {
	// 1. apply Gaussian blur to combat noise
	Mat blurredFrame;
	GaussianBlur(frame, blurredFrame, Size(5,5), 0);
	imwrite("Colours/frame.jpg", frame);
	imwrite("Colours/blurred.jpg", blurredFrame);

	// 2. convert from BGR to HSV colour frame
	Mat frameHSV;
	cvtColor(blurredFrame, frameHSV, COLOR_BGR2HSV);

	// 3. threshold the red values to white, rest to black
	Mat frameThresh1;
	inRange(frameHSV, Scalar(0, 70, 50), Scalar(15, 255, 255), frameThresh1);

	Mat frameThresh2;
	inRange(frameHSV, Scalar(165, 70, 50), Scalar(180, 255, 255), frameThresh2);

	Mat frameThresh = frameThresh1 | frameThresh2;

	imwrite("Colours/red.jpg", frameThresh);

	// 4. count the values above the threshold to find the proportion of red pixels 
	int numRed = 0;
	for (int y=0; y < frameThresh.rows; y++) {
		for (int x=0; x < frameThresh.cols; x++) {
			if (frameThresh.at<uchar>(y,x) == 255) numRed++;
		}
	}
	float proportion = (float) numRed / (float) (frameThresh.rows * frameThresh.cols);
	
	// 5. return true if there is a larger amount of red in the image
	if (proportion > threshold) return true;
	else return false;
}

bool minorityWhite(Mat frame, float threshold) {
	// 1. blur the image to reduce noise
	Mat blurredFrame;
	GaussianBlur(frame, blurredFrame, Size(5,5), 0);

	// 2. convert from BGR to HSV colour space
	Mat frameHSV;
	cvtColor(blurredFrame, frameHSV, COLOR_BGR2HSV);

	// 3. threshold the white values
	Mat frameThresh;
	inRange(frameHSV, Scalar(0, 0, 50), Scalar(180, 70, 255), frameThresh);

	imwrite("Colours/white.jpg", frameThresh);

	// 4. count the values above the threshold to find the proportion of white pixels 
	int numWhite = 0;
	for (int y=0; y < frameThresh.rows; y++) {
		for (int x=0; x < frameThresh.cols; x++) {
			if (frameThresh.at<uchar>(y,x) == 255) numWhite++;
		}
	}
	float proportion = (float) numWhite / (float) (frameThresh.rows * frameThresh.cols);
	
	// 5. return true if there is a small amount of white in the image
	if (proportion > 0.05 && proportion < threshold) return true;
	else return false;
}
