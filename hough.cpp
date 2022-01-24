#include "hough.h"

using namespace cv;

// --------------------HOUGH TRANSFORM FUNCTIONS--------------------

// apply the kernel to the input image using convolution
void sobel(cv::Mat &input, cv::Mat &output, int kData[], int rows, int cols) {
    // initialise the output using the input
    output.create(input.size(), CV_32F);

    // create kernel
    cv::Mat kernel = cv::Mat(rows, cols, CV_32S, kData);

    // create a padded version of the input
    int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;
    cv::Mat paddedInput;
    cv::copyMakeBorder( input, paddedInput,
                        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                        cv::BORDER_REPLICATE );

    // now we can do the convolution
    for ( int j = 0; j < input.rows; j++ )
    {
        for( int i = 0; i < input.cols; i++ )
        {
            double sum = 0.0;
            for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
            {
                for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
                {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = ( int ) paddedInput.at<uchar>( imagey, imagex );
                    int kernelval = kernel.at<int>( kernely, kernelx );

                    // do the multiplication
                    sum += imageval * kernelval;
                }
            }
            // set the output as the sum of convolution
            output.at<float>(j, i) = (float) sum;
        }
    }
}

// given the derivative matrices for the x and y direction, produce the magnitude and direction matrices
float findDerivative(cv::Mat &xInput, cv::Mat &yInput, cv::Mat &magnitude, cv::Mat &direction) {
    assert(xInput.size() == yInput.size());

    // initialise the output using the input
    magnitude.create(xInput.size(), xInput.type());
    direction.create(xInput.size(), xInput.type());

    // keep track of the largest magnitude encountered
    float maxMag = 0;

    // iterate over the rows and columns to calculate magnitude and direction at each point
    for ( int j = 0; j < xInput.rows; j++ )
    {
        for( int i = 0; i < xInput.cols; i++ )
        {
            // get the values for the x and y derivatives
            float xVal = xInput.at<float>(j, i);
            float yVal = yInput.at<float>(j, i);

            // find the magnitude and direction
            float mag = sqrt(pow(xVal, 2) + pow(yVal, 2));
            float dir;
            if (xVal == 0) dir = M_PI / 2;
            else dir = atan(yVal / xVal);

            // set the output value as the sum of the convolution
            magnitude.at<float>(j, i) = mag;
            direction.at<float>(j, i) = dir;

            // update largest magnitude if necessary
            if (mag > maxMag) maxMag = mag;
        }
    }

    return maxMag;
}

// set all values above the threshold to 255 and crush the rest to 0
void threshold(cv::Mat &input, cv::Mat &output, int threshold) {
    // initialise the output using the input
    output.create(input.size(), CV_8U);

    // iterate over the rows and columns to threshold each pixel value
    for ( int j = 0; j < input.rows; j++ )
    {
        for( int i = 0; i < input.cols; i++ )
        {
            float pixelVal = input.at<float>(j, i);
            if (pixelVal < threshold) output.at<uchar>(j, i) = (uchar) 0;
            else output.at<uchar>(j, i) = (uchar) 255;
        }
    }
}

// create the circular Hough space, setting all entries to 0
void initialiseHough(std::vector<std::vector<std::vector<int> > > &houghSpace, int rows, int cols, int numRadii) {
    for (int j=0; j < rows; j++) {
        std::vector<std::vector<int> > row;
        for (int i=0; i < cols; i++) {
            std::vector<int> col;
            for (int k=0; k < numRadii; k++) {
                col.push_back(0);
            }
            row.push_back(col);
        }
        houghSpace.push_back(row);
    }
}

// populate the circular Hough space, using the thresholded magnitudes and orientation matrix
int fillHough(cv::Mat &thresholdedMagnitudes, cv::Mat &orientation, int threshold, int minRadius,
              std::vector<std::vector<std::vector<int> > > &houghSpace, int rows, int cols, int radii) {
    // keep track of the largest hough entry encountered
    int maxHough = 0;
    for (int y=0; y < rows; y++) {
        for (int x=0; x < cols; x++) {
            int mag = (int) thresholdedMagnitudes.at<uchar>(y,x);
            if (mag == 255) {
                float dir = orientation.at<float>(y,x);
                for (int r = 0; r < radii; r++) {
                    int posX = (int) x + (minRadius + r) * cos(dir);
                    int posY = (int) y + (minRadius + r) * sin(dir);
                    int negX = (int) x - (minRadius + r) * cos(dir);
                    int negY = (int) y - (minRadius + r) * sin(dir);
                    if (posX >= 0 and posX < cols and posY >= 0 and posY < rows) {
                        houghSpace.at(posY).at(posX).at(r) += 1;
                        maxHough = max(maxHough, houghSpace.at(posY).at(posX).at(r));
                    }
                    if (negX >= 0 and negX < cols and negY >= 0 and negY < rows) {
                        houghSpace.at(negY).at(negX).at(r) += 1;
                        maxHough = max(maxHough, houghSpace.at(negY).at(negX).at(r));
                    }
                }
            }
        }
    }
    return maxHough;
}

// set all values in the Hough space above the threshold to 1 and the rest to 0
void thresholdHough(std::vector<std::vector<std::vector<int> > > &houghSpace, int rows, int cols, int radii, int threshold) {
    for (int j=0; j < rows; j++) {
        for (int i=0; i < cols; i++) {
            for (int r=0; r < radii; r++) {
                houghSpace.at(j).at(i).at(r) = (houghSpace.at(j).at(i).at(r) > threshold) ? 1 : 0;
            }
        }
    }
}

// given the thresholded Hough space, return a vector of circles (including position and radius)
int findCircles(std::vector<std::vector<int> > &circles, std::vector<std::vector<std::vector<int> > > &houghSpace, 
                 int rows, int cols, int radii, int minRadius) {
    // each average circle is stored as a vector of the form {x, y, r, n}
    // where (x,y) is the centre, r is the radius and n is the number of circles that belong to this class

    // this represents the minimum distance allowed between two distinct circle centres
    const int MIN_DIST = minRadius + radii;

    // classify each distinct circle as the average of nearby circles
    for (int j=0; j < rows; j++) {
        for (int i=0; i < cols; i++) {
            // find the average radius for the centre point (i, j)
            int radiusSum = 0;
            int significantNum = 0;
            for (int r=0; r < radii; r++) {
                if (houghSpace.at(j).at(i).at(r) == 1) {
                    radiusSum += r;
                    significantNum++;
                }
            }
            if (significantNum > 0) {
                int avgRadius = radiusSum / significantNum;
                // if there are no existing circles, append it
                if (circles.size() == 0) {
                    std::vector<int> circle;
                    circle.push_back(i);
                    circle.push_back(j);
                    circle.push_back(avgRadius);
                    circle.push_back(1);
                    circles.push_back(circle);
                }
                // if there are existing circles, iterate over each of them until finding a 'match' or reaching the end of the vector
                else {
                    bool match = false;
                    int k = 0;
                    while (k < circles.size() && !match) {
                        // find the Euclidean distance between their centres
                        int xDiff = i - circles[k][0];
                        int yDiff = j - circles[k][1];
                        int dist = sqrt(pow(xDiff, 2) + pow(yDiff, 2));
                        // if this is less than the threshold, they are considered to belong to the same circle and therefore 'match'
                        if (dist < MIN_DIST) {
                            // update this circle in the vector with the new average values
                            int n = circles[k][3];
                            circles[k][0] = (circles[k][0] * n + i) / (n + 1);
                            circles[k][1] = (circles[k][1] * n + j) / (n + 1);
                            circles[k][2] = (circles[k][2] * n + avgRadius) / (n + 1);
                            circles[k][3]++;
                            match = true;
                        }
                        k++;
                    }
                    // if there are no matches, this must be a new circle class, so append it to the vector
                    if (!match) {
                        std::vector<int> circle;
                        circle.push_back(i);
                        circle.push_back(j);
                        circle.push_back(avgRadius);
                        circle.push_back(1);
                        circles.push_back(circle);
                    }
                }
            }
        }
    }
    return circles.size();
}

// --------------------DISPLAY FUNCTIONS--------------------

void displayGradient(cv::Mat &input, std::string filename) {
    // create the output image
    cv::Mat output = cv::Mat(input.size(), CV_8U);

    // find the largest gradient in the input matrix
    double minVal, maxVal;
    cv::minMaxLoc(input, &minVal, &maxVal);

    // iterate over each value in the input matrix
    for (int j=0; j < input.rows; j++) {
        for (int i=0; i < input.cols; i++) {
            // normalise the value, mapping it to a greyscale pixel between 0 and 255
            float gradient = input.at<float>(j, i);
            output.at<uchar>(j, i) = (uchar) (255 * (abs(gradient) / maxVal));
        }
    }

    // write the matrix to a file
    imwrite(filename, output);
}

void displayDirection(cv::Mat &input, std::string filename) {
    // create the output image
    cv::Mat output = cv::Mat(input.size(), CV_8U);

    // iterate over each value in the input matrix
    for (int j=0; j < input.rows; j++) {
        for (int i=0; i < input.cols; i++) {
            // normalise the value, mapping it to a greyscale pixel between 0 and 255
            float direction = input.at<float>(j, i);
            output.at<uchar>(j, i) = (uchar) ((direction + (M_PI / 2)) / M_PI * 255);
        }
    }

    // write the matrix to a file
    imwrite(filename, output);
}

void displayHough(std::vector<std::vector<std::vector<int> > > &houghSpace, int rows, int cols, int radii, string filename) {
    // find the max summed radius value
    int maxSum = 0;
    for (int j=0; j < rows; j++) {
        for (int i=0; i < cols; i++) {
            int radiusSum = 0;
            for (int r=0; r < radii; r++) {
                radiusSum += houghSpace.at(j).at(i).at(r);
            }
            if (radiusSum > maxSum) maxSum = radiusSum;
        }
    }
    // create a visualisation of the summed radii, normalising with maxSum
    cv::Mat output = cv::Mat(rows, cols, CV_8U);
    for (int j=0; j < rows; j++) {
        for (int i=0; i < cols; i++) {
            int sum = 0;
            for (int r=0; r < radii; r++) {
                sum += houghSpace.at(j).at(i).at(r);
            }
            int normalisedSum = (int) 255 * (sum / maxSum);
            // output.at<uchar>(j,i) = (uchar) normalisedSum;
            output.at<uchar>(j,i) = (uchar) (255 * sum / maxSum);
        }
    }
    // write the matrix to a file
    imwrite(filename, output);
}

void displayCircles(std::vector<std::vector<int> > &circles, cv::Mat &image, int minRadius, string filename) {
    cv::Mat output = image.clone();

    // draw each circle over a copy of the original image
    for (int i=0; i < circles.size(); i++) {
        int x = circles[i][0];
        int y = circles[i][1];
        int r = circles[i][2];
        circle(output, Point(x,y), minRadius + r, Scalar(255,0,0), 2);
    }

    // write the matrix to a file
    imwrite(filename, output);
}

// --------------------MAIN DETECTION FUNCTION--------------------

void houghCircles(const char *imageName, std::vector<std::vector<int> > &circles, int magThresh, int minRadius, int numRadii, int houghThresh) {
    const string filepath = "Hough/";

    // load the input image
    std::cout << "imageName: " << imageName << std::endl;
    Mat image;
    image = imread( imageName, 1 );
    if(!image.data) {
        printf( " No image data \n " );
        exit(-1);
    }

    // set the threshold constants
    const int MAG_THRESH = magThresh;
    const int MIN_RADIUS = minRadius;
    const int NUM_RADII = numRadii;
    const int HOUGH_THRESH = houghThresh;

    // convert image to greyscale
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    // initialise the kernel matrices
    int kX[9] = {-1, 0, 1, 
                 -2, 0, 2, 
                 -1, 0, 1};

    int kY[9] = {-1, -2, -1, 
                  0,  0,  0, 
                  1,  2,  1};

    // find the derivatives matrices
    Mat xDerivatives;
    sobel(gray_image, xDerivatives, kX, 3, 3);
    // displayGradient(xDerivatives, filepath + "xDerivatives.jpg");

    Mat yDerivatives;
    sobel(gray_image, yDerivatives, kY, 3, 3);
    // displayGradient(yDerivatives, filepath + "yDerivatives.jpg");

    // find the magnitude and direction matrices
    Mat magnitudes;
    Mat direction;
    float maxMag = findDerivative(xDerivatives, yDerivatives, magnitudes, direction);
    // displayGradient(magnitudes, filepath + "magnitude.jpg");
    // displayDirection(direction, filepath + "direction.jpg");
    std::cout << "Max magnitude = " << maxMag << std::endl;

    // threshold the magnitudes
    Mat thresholdedMagnitudes;
    threshold(magnitudes, thresholdedMagnitudes, MAG_THRESH);
    imwrite(filepath + "thresholdedMagnitudes.jpg", thresholdedMagnitudes);

    // create the Hough space
    std::vector<std::vector<std::vector<int> > > houghSpace;
    initialiseHough(houghSpace, gray_image.rows, gray_image.cols, NUM_RADII);
    int maxHough = fillHough(thresholdedMagnitudes, direction, HOUGH_THRESH, MIN_RADIUS, houghSpace, gray_image.rows, gray_image.cols, NUM_RADII);
    std::cout << "Max value in Hough space = " << maxHough << std::endl;
    // displayHough(houghSpace, gray_image.rows, gray_image.cols, NUM_RADII, filepath + "houghSpace.jpg");

    // threshold the Hough space
    Mat thresholdedHough;
    thresholdHough(houghSpace, gray_image.rows, gray_image.cols, NUM_RADII, HOUGH_THRESH);
    displayHough(houghSpace, gray_image.rows, gray_image.cols, NUM_RADII, filepath + "thresholdedHough.jpg");

    // display the detected circles overlaid on the original image
    // int numCircles = displayCircles(image, houghSpace, gray_image.rows, gray_image.cols, NUM_RADII, MIN_RADIUS, "detectedCircles.jpg");
    int numCircles = findCircles(circles, houghSpace, gray_image.rows, gray_image.cols, NUM_RADII, MIN_RADIUS);
    displayCircles(circles, image, MIN_RADIUS, "detectedCircles.jpg");
    std::cout << "Number of detected circles = " << numCircles << std::endl;
}
