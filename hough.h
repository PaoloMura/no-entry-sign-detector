#include <array>
#include <stdio.h>
#include <string>
#include <cmath>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

void houghCircles(const char *imageName, std::vector<std::vector<int> > &circles, int magThresh, int minRadius, int numRadii, int houghThresh);
