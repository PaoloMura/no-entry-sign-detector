#include <array>
#include <stdio.h>
#include <string>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

using namespace cv;


class HoughTransform {
    protected:
        // matrices
        Mat image;
        Mat gray_image;
        Mat xDerivatives;
        Mat yDerivatives;
        Mat magnitudes;
        Mat direction;
        Mat thresholdedMagnitudes;
        Mat thresholdedHough;

        // convolution kernels
        int kX[9] = {-1, 0, 1, 
                     -2, 0, 2, 
                     -1, 0, 1};

        int kY[9] = {-1, -2, -1, 
                      0,  0,  0, 
                      1,  2,  1};

        // constants (some are not actual constants since they need to be initialised at runtime)
        const string filepath = "Hough/";
        int MAG_THRESH;
        int MIN_RADIUS;
        int NUM_RADII;
        int HOUGH_THRESH;

        // methods
        void sobel();
        int findDerivative();
        void threshold();
        void populateDerivativeMatrices(const char *filename, string fileNum);
        void displayGradient(cv::Mat &input, std::string filename);
        void displayDirection(cv::Mat &input, std::string filename);

    public:
        virtual void performTransform(const char *filename, string fileNum) = 0;
};


class MyHoughLines: public HoughTransform {
    private:
        std::vector<std::vector<int> > houghSpace;
        const int NUM_ANGLES = 360;
        const int SMALL_THETA = 5;
        const int SMALL_RHO = 10;
        int maxDist;

        void initialiseHough();
        int fillHough();
        void thresholdHough();
        int findLines();

        void displayHough(int maxHough, string filename);
        void displayLines(string filename);

    public:
        // store the result as a publicly accessible attribute
        std::vector<std::vector<float> > lines;
        
        MyHoughLines(int magThresh, int houghThresh);
        void performTransform(const char*filename, string fileNum);
};


class MyHoughCircles: public HoughTransform {
    private:
        std::vector<std::vector<std::vector<int> > > houghSpace;

        void initialiseHough();
        int fillHough();
        void thresholdHough();
        int findCircles();

        void displayHough(string filename);
        void displayCircles(string filename);

    public:
        // store the result as a publicly accessible attribute
        std::vector<std::vector<int> > circles;

        MyHoughCircles(int magThresh, int minRadius, int numRadii, int houghThresh);
        void performTransform(const char *filename, string fileNum);
};
