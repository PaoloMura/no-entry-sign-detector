#include "houghOOP.h"

using namespace cv;

// -------------------- CONSTRUCTORS -------------------- //

MyHoughCircles::MyHoughCircles(int magThresh, int minRadius, int numRadii, int houghThresh) {
    MAG_THRESH = magThresh;
    MIN_RADIUS = minRadius;
    NUM_RADII = numRadii;
    HOUGH_THRESH = houghThresh;
};

MyHoughLines::MyHoughLines(int magThresh, int houghThresh) {
    MAG_THRESH = magThresh;
    HOUGH_THRESH = houghThresh;
}


// -------------------- HOUGH TRANSFORM METHODS -------------------- //

void HoughTransform::sobel() {
    // initialise the output matrices
    xDerivatives.create(gray_image.size(), CV_32F);
    yDerivatives.create(gray_image.size(), CV_32F);

    // create kernels
    cv::Mat kernelDX = cv::Mat(3, 3, CV_32S, kX);
    cv::Mat kernelDY = cv::Mat(3, 3, CV_32S, kY);

    // create a padded version of the input
    int kernelRadiusX = ( kernelDX.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernelDX.size[1] - 1 ) / 2;
    cv::Mat paddedInput;
    cv::copyMakeBorder( gray_image, paddedInput,
                        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                        cv::BORDER_REPLICATE );

    // now we can do the convolution
    for (int j = 0; j < gray_image.rows; j++)
    {
        for (int i = 0; i < gray_image.cols; i++)
        {
            double sumX = 0;
            double sumY = 0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
            {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
                {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernels
                    int imageval = ( int ) paddedInput.at<uchar>( imagey, imagex );
                    int kernelDXval = kernelDX.at<int>(kernely, kernelx);
                    int kernelDYval = kernelDY.at<int>(kernely, kernelx);

                    // do the multiplication
                    sumX += imageval * kernelDXval;
                    sumY += imageval * kernelDYval;
                }
            }
            // set the output as the sum of convolution
            xDerivatives.at<float>(j, i) = (float) sumX;
            yDerivatives.at<float>(j, i) = (float) sumY;
        }
    }
}

int HoughTransform::findDerivative() {
    assert(xDerivatives.size() == yDerivatives.size());

    // initialise the output using the input
    magnitudes.create(xDerivatives.size(), xDerivatives.type());
    direction.create(xDerivatives.size(), xDerivatives.type());

    // keep track of the largest magnitude encountered
    float maxMag = 0;

    // iterate over the rows and columns to calculate magnitude and direction at each point
    for ( int j = 0; j < xDerivatives.rows; j++ )
    {
        for( int i = 0; i < xDerivatives.cols; i++ )
        {
            // get the values for the x and y derivatives
            float xVal = xDerivatives.at<float>(j, i);
            float yVal = yDerivatives.at<float>(j, i);

            // find the magnitude and direction
            float mag = sqrt(pow(xVal, 2) + pow(yVal, 2));
            float dir;
            if (xVal == 0) dir = M_PI / 2;
            else dir = atan2(yVal, xVal);

            // set the output value as the sum of the convolution
            magnitudes.at<float>(j, i) = mag;
            direction.at<float>(j, i) = dir;

            // update largest magnitude if necessary
            if (mag > maxMag) maxMag = mag;
        }
    }

    return maxMag;
}

void HoughTransform::threshold() {
    // initialise the output using the input
    thresholdedMagnitudes.create(magnitudes.size(), CV_8U);

    // iterate over the rows and columns to threshold each pixel value
    for ( int j = 0; j < magnitudes.rows; j++ )
    {
        for( int i = 0; i < magnitudes.cols; i++ )
        {
            float pixelVal = magnitudes.at<float>(j, i);
            if (pixelVal < MAG_THRESH) thresholdedMagnitudes.at<uchar>(j, i) = (uchar) 0;
            else thresholdedMagnitudes.at<uchar>(j, i) = (uchar) 255;
        }
    }
}

void HoughTransform::populateDerivativeMatrices(const char *filename, string fileNum) {
    // load the input image
    image = imread(filename, 1);
    if(!image.data) {
        printf("No image data \n");
        exit(-1);
    }

    // convert the image to greyscale
    cvtColor( image, gray_image, CV_BGR2GRAY );

    // compute the magnitude and direction of the gradient images
    sobel();
    float maxMag = findDerivative();
    MAG_THRESH = 0.37 * maxMag;
    printf("Max magnitude: %d \n", maxMag);
    printf("Gradient Magnitude threshold used: %d \n", MAG_THRESH);
    
    threshold();
}

void HoughTransform::displayGradient(cv::Mat &input, std::string filename) {
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

void HoughTransform::displayDirection(cv::Mat &input, std::string filename) {
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


// -------------------- HOUGH LINES METHODS -------------------- //

void MyHoughLines::initialiseHough() {
    // set the max distance as the diagonal from top left to bottom right corner
    maxDist = sqrt(pow(gray_image.rows, 2) + pow(gray_image.cols, 2));
    for (int r=0; r < maxDist; r++) {
        std::vector<int> row;
        for (int t=0; t < NUM_ANGLES; t++) {
            row.push_back(0);
        }
        houghSpace.push_back(row);
    }
}

int MyHoughLines::fillHough() {
    int maxHough = 0;
    for (int y = 0; y < gray_image.rows; y++) {
        for (int x = 0; x < gray_image.cols; x++) {
            int mag = (int) thresholdedMagnitudes.at<uchar>(y,x);
            if (mag == 255) {
                for (int t=0; t < NUM_ANGLES; t++) {
                    float theta = t * (M_PI / 180.0f);
                    int r = round(abs(x * cos(theta) + y * sin(theta)));
                    if (r >= 0 && r < houghSpace.size()) {
                        houghSpace[r][t] += 1;
                        maxHough = max(maxHough, houghSpace[r][t]);
                    }
                }


                // // extract the direction angle and map it from (-PI/2 - PI/2) to (0 - 180)
                // float dir = direction.at<float>(y,x);
                // int dirInt = (dir + (M_PI / 2)) * (180 / M_PI);
                // int t0 = max(dirInt - SMALL_THETA, 0);
                // int t1 = min(dirInt + SMALL_THETA, NUM_ANGLES);
                // for (int t = t0; t < t1; t++) {
                //     // convert the table theta to its radian angle
                //     float theta = (t == 0) ? M_PI / 2 : t * (M_PI / 180) - (M_PI / 2);
                //     // calculate the distance rho and update the Hough space
                //     int r = (int) abs(x * cos(theta) + y * sin(theta));
                //     houghSpace[r][t] += 1;
                //     maxHough = max(maxHough, houghSpace[r][t]);
                // }
            }
        }
    }
    return maxHough;
}

void MyHoughLines::thresholdHough() {
    for (int r=0; r < maxDist; r++) {
        for (int t=0; t < NUM_ANGLES; t++) {
            houghSpace.at(r).at(t) = (houghSpace.at(r).at(t) > HOUGH_THRESH) ? 1 : 0;
        }
    }
}

int MyHoughLines::findLines() {
    // return a vector of lines
    // each line consists of (r, t, n) where 
    // r is the perpendicular distance from origin to line;
    // t is the mean angle between the line's normal and the x-axis; and
    // n is the number of lines in this class
    for (int r=0; r < maxDist; r++) {
        for (int t=0; t < NUM_ANGLES; t++) {      
            if (houghSpace[r][t] != 0) {
                float theta = t * (M_PI / 180.0f);
                std::vector<float> line;
                line.push_back(r);
                line.push_back(theta);
                line.push_back(1);
                lines.push_back(line);
            }

            // // convert the angle to radians
            // float theta = t * (M_PI / 180) - (M_PI / 2);
            // // if there are no existing lines, append this as a new line
            // if (lines.size() == 0) {
            //     std::vector<float> line;
            //     line.push_back(r);
            //     line.push_back(theta);
            //     line.push_back(1);
            //     lines.push_back(line);
            // }
            // // otherwise, check if there is a 'match'
            // else {
            //     bool match = false;
            //     int index = 0;
            //     while (!match && index < lines.size()) {
            //         // if two lines have a similar rho and angle, they 'match' (i.e. belong to the same class)
            //         float r1 = lines[index][0];
            //         float t1 = lines[index][1];
            //         float n = lines[index][2];
            //         if (abs(r - r1) < SMALL_RHO && abs(t - t1) < SMALL_THETA) {
            //             // update the mean angle of this line
            //             float t2 = (t1 * n + theta) / (n + 1);
            //             lines[index][1] = t2;
            //             lines[index][2]++;
            //             match = true;
            //         }
            //         index++;
            //     }
            //     // if there are no matches, append this as a new line
            //     if (!match) {
            //         std::vector<float> line;
            //         line.push_back(r);
            //         line.push_back(theta);
            //         line.push_back(1);
            //         lines.push_back(line);
            //     }
            // }
        }
    }
    return lines.size();
}

void MyHoughLines::performTransform(const char *filename, string fileNum) {
    // this loads the image and computes the magnitude and direction of the gradients
    populateDerivativeMatrices(filename, fileNum);

    // TODO: remove this
    displayGradient(xDerivatives, "Hough/xDerivatives.jpg");
    displayGradient(yDerivatives, "Hough/yDerivatives.jpg");
    displayGradient(magnitudes, "Hough/magnitudes.jpg");
    displayDirection(direction, "Hough/direction.jpg");
    imwrite("Hough/thresholdedMagnitudes.jpg", thresholdedMagnitudes);

    // create the Hough space
    initialiseHough();

    int maxHough = fillHough();
    printf("Max value in line Hough space = %d \n", maxHough);

    // TODO: remove this
    displayHough(maxHough, "Hough/houghLines.jpg");

    thresholdHough();

    // TODO: remove this
    displayHough(1, "Hough/thresholdedHoughLines.jpg");

    // find the lines in the image
    int numLines = findLines();
    printf("Number of detected lines = %d \n", numLines);

    // TODO: remove this
    displayLines("Hough/detectedLines.jpg");
}

void MyHoughLines::displayHough(int maxHough, string filename) {
    int rows = houghSpace.size();
    int cols = houghSpace[0].size();
    cv::Mat output = cv::Mat(rows, cols, CV_8U);
    for (int j=0; j < rows; j++) {
        for (int i=0; i < cols; i++) {
            int value = houghSpace[j][i];
            int normalisedValue = (int) (255 * (value / maxHough));
            output.at<uchar>(j,i) = (uchar) normalisedValue;
        }
    }
    // write the matrix to a file
    imwrite(filename, output);
}

void MyHoughLines::displayLines(string filename) {
    cv::Mat output = image.clone();

    // draw each line over a copy of the original image
    for (int i=0; i < lines.size(); i++) {
        int r = lines[i][0];
        int t = lines[i][1];

        cv::Point p1;
        cv::Point p2;

        p1.x = r * cos(t) - 1000 * sin(t);
        p1.y = r * sin(t) + 1000 * cos(t);
        p2.x = r * cos(t) + 1000 * sin(t);
        p2.y = r * sin(t) - 1000 * cos(t);

        line(output, p1, p2, cv::Scalar(255, 0, 0), 2);
    }

    // write the matrix to a file
    imwrite(filename, output);
}


// -------------------- HOUGH CIRCLES METHODS -------------------- //

void MyHoughCircles::initialiseHough() {
    for (int j=0; j < gray_image.rows; j++) {
        std::vector<std::vector<int> > row;
        for (int i=0; i < gray_image.cols; i++) {
            std::vector<int> col;
            for (int k=0; k < NUM_RADII; k++) {
                col.push_back(0);
            }
            row.push_back(col);
        }
        houghSpace.push_back(row);
    }
}

int MyHoughCircles::fillHough() {
    // keep track of the largest hough entry encountered
    int maxHough = 0;
    for (int y=0; y < gray_image.rows; y++) {
        for (int x=0; x < gray_image.cols; x++) {
            int mag = (int) thresholdedMagnitudes.at<uchar>(y,x);
            if (mag == 255) {
                float dir = direction.at<float>(y,x);
                for (int r = 0; r < NUM_RADII; r++) {
                    int posX = (int) x + (MIN_RADIUS + r) * cos(dir);
                    int posY = (int) y + (MIN_RADIUS + r) * sin(dir);
                    int negX = (int) x - (MIN_RADIUS + r) * cos(dir);
                    int negY = (int) y - (MIN_RADIUS + r) * sin(dir);
                    if (posX >= 0 and posX < gray_image.cols and posY >= 0 and posY < gray_image.rows) {
                        houghSpace.at(posY).at(posX).at(r) += 1;
                        maxHough = max(maxHough, houghSpace.at(posY).at(posX).at(r));
                    }
                    if (negX >= 0 and negX < gray_image.cols and negY >= 0 and negY < gray_image.rows) {
                        houghSpace.at(negY).at(negX).at(r) += 1;
                        maxHough = max(maxHough, houghSpace.at(negY).at(negX).at(r));
                    }
                }
            }
        }
    }
    return maxHough;
}

void MyHoughCircles::thresholdHough() {
    for (int j=0; j < gray_image.rows; j++) {
        for (int i=0; i < gray_image.cols; i++) {
            for (int r=0; r < NUM_RADII; r++) {
                houghSpace.at(j).at(i).at(r) = (houghSpace.at(j).at(i).at(r) > HOUGH_THRESH) ? 1 : 0;
            }
        }
    }
}

int MyHoughCircles::findCircles() {
    // each average circle is stored as a vector of the form {x, y, r, n}
    // where (x,y) is the centre, r is the radius and n is the number of circles that belong to this class

    // this represents the minimum distance allowed between two distinct circle centres
    const int MIN_DIST = MIN_RADIUS + NUM_RADII;

    // classify each distinct circle as the average of nearby circles
    for (int j=0; j < gray_image.rows; j++) {
        for (int i=0; i < gray_image.cols; i++) {
            // find the average radius for the centre point (i, j)
            int radiusSum = 0;
            int significantNum = 0;
            for (int r=0; r < NUM_RADII; r++) {
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
    // add the minimum radius to each chosen circle
    for (int i=0; i < circles.size(); i++) {
        circles[i][2] += MIN_RADIUS;
    }
    return circles.size();
}

void MyHoughCircles::performTransform(const char *filename, string fileNum) {
    // this loads the image and computes the magnitude and direction of the gradients
    populateDerivativeMatrices(filename, fileNum);

    imwrite("MagnitudeThresholds/image" + fileNum + ".jpg", thresholdedMagnitudes);

    // create the Hough space
    initialiseHough();
    int maxHough = fillHough();
    HOUGH_THRESH = 0.55 * maxHough;
    printf("Max value in circle Hough space = %d \n", maxHough);
    printf("Hough threshold used: %d \n", HOUGH_THRESH);

    // TODO: remove this
    displayHough("CircleHoughSpaces/image" + fileNum + ".jpg");

    thresholdHough();
    
    // find the circles in the image
    int numCircles = findCircles();
    printf("Number of detected circles = %d \n", numCircles);

    // TODO: remove this
    displayHough("Hough/thresholdedHoughCircles.jpg");
    displayCircles("Hough/detectedCircles.jpg");
}

void MyHoughCircles::displayHough(string filename) {
    // find the max summed radius value
    int maxSum = 0;
    for (int j=0; j < gray_image.rows; j++) {
        for (int i=0; i < gray_image.cols; i++) {
            int radiusSum = 0;
            for (int r=0; r < NUM_RADII; r++) {
                radiusSum += houghSpace.at(j).at(i).at(r);
            }
            if (radiusSum > maxSum) maxSum = radiusSum;
        }
    }
    // create a visualisation of the summed radii, normalising with maxSum
    cv::Mat output = cv::Mat(gray_image.rows, gray_image.cols, CV_8U);
    for (int j=0; j < gray_image.rows; j++) {
        for (int i=0; i < gray_image.cols; i++) {
            int sum = 0;
            for (int r=0; r < NUM_RADII; r++) {
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

void MyHoughCircles::displayCircles(string filename) {
    cv::Mat output = image.clone();

    // draw each circle over a copy of the original image
    for (int i=0; i < circles.size(); i++) {
        int x = circles[i][0];
        int y = circles[i][1];
        int r = circles[i][2];
        circle(output, Point(x,y), r, Scalar(255,0,0), 2);
    }

    // write the matrix to a file
    imwrite(filename, output);
}
