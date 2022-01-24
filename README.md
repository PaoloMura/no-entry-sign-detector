# Background

This repo is my coursework submission for the year 3 Image Processing & Computer Vision unit. The task description can be found in the `COMS30031_2021_CW.pdf` file. In summary it involved three subtasks:

1. Train a Viola-Jones feature detector to detect front-facing human faces in an image
2. Adapt the code to detect "no entry" signs
3. Integrate this method with your own implementation of the Hough Transform shape detector algorithm
4. Combine these approaches with a third method of your own choice

# Result

I was able to complete all four subtasks. My final solution had a true positive rate (TPR) of ~0.6 and an F1-Score of ~0.7. It made very few false detections but there were some signs that were too small, occluded or under challenging lighting conditions that it was unable to detect.

A detailed discussion of my method, rationale and analysis can be found in `report.pdf`.

# How to Run

## To run faces.cpp (face detection on a given image):

`make face` or just `make` followed by
`./a.out No_entry/NoEntry1.bmp Face_ground_truth/NoEntry1.csv`
where 1 can be replaced by the number of the image file you wish to process.

## To run noEntry.cpp ("no entry" sign detection on a given image):

Within the `detectAndDisplay` procedure in the `noEntry.cpp` file:
* If you just want to run Viola-Jones, comment out sections 5, 6, 7 and 8.
* If you want to run Viola-Jones with the Hough Circle Detector, comment out sections 4, 7 and 8.
* If you want to run all three (Viola-Jones, Hough Circles and Colour Filter), comment out sections 4 and 6.

Then, in the terminal run
`make noEntry` followed by
`./a.out 1` where 1 can be replaced by the number of the image file you wish to process.
