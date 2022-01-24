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

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, const char *filename );
void parseCSV(const char *filename, vector<Rect> &positiveFaces);
int countSuccess(std::vector<Rect> foundFaces, std::vector<Rect> positiveFaces);
bool intersects(Rect a, Rect b);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

/** Global constants */
const float UOI_THRESHOLD = 0.5;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	const char *filename = argv[2];

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, filename );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, const char *filename )
{
	std::vector<Rect> faces;
	std::vector<Rect> positiveFaces;
	parseCSV(filename, positiveFaces);
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.01, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(500,500) );

       // 3. Print number of faces found
	std::cout << "Number of detected regions: " << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	   // 5. Draw box around positive faces
	for (int i=0; i < positiveFaces.size(); i++) {
		rectangle(frame, Point(positiveFaces[i].x, positiveFaces[i].y), Point(positiveFaces[i].x + positiveFaces[i].width, positiveFaces[i].y + positiveFaces[i].height), Scalar( 0, 0, 255 ), 2);
	}

	   // 6. summarise the success rate
	float truePositives = countSuccess(faces, positiveFaces);
	float numPositive = positiveFaces.size();
    float numDetected = faces.size();
	float precision = truePositives / numDetected;
	float recall = truePositives / numPositive;
	float f1Score = 2 / ((1 / precision) + (1 / recall));

	cout << "TPR: " << recall << endl;
	cout << "F1 Score: " << f1Score << endl;
}

int countSuccess(std::vector<Rect> foundFaces, std::vector<Rect> positiveFaces) {
	int numOfSuccess = 0;
	for (int i=0; i < foundFaces.size(); i++) {
		Rect foundFace = foundFaces[i];
		float foundArea = foundFace.area();
		for (int j=0; j < positiveFaces.size(); j++) {
			Rect positiveFace = positiveFaces[j];
			float positiveArea = positiveFace.area();
			Rect intersection = foundFace & positiveFace;
			float intersectionArea = intersection.area();
			if (intersection.area() != 0) {
				float uoi = intersectionArea / (foundArea + positiveArea - intersectionArea);
				if (uoi > UOI_THRESHOLD) numOfSuccess++;
			}
		}
	}
	return numOfSuccess;
}

/** @function parseCSV */
void parseCSV(const char *filename, vector<Rect> &positiveFaces) {
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
		Rect positiveFace = Rect(values[0], values[1], values[2], values[3]);
		positiveFaces.push_back(positiveFace);
	}
}
