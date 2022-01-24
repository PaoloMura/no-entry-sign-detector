To run faces.cpp:
`make face` or just `make` followed by
`./a.out No_entry/NoEntry1.bmp Face_ground_truth/NoEntry1.csv`
where 1 can be replaced by the number of the image file you wish to process.



To run noEntry.cpp:

Within the `detectAndDisplay` procedure in the `noEntry.cpp` file:
* If you just want to run Viola-Jones, comment out sections 5, 6, 7 and 8.
* If you want to run Viola-Jones with the Hough Circle Detector, comment out sections 4, 7 and 8.
* If you want to run all three (Viola-Jones, Hough Circles and Colour Filter), comment out sections 4 and 6.

Then, in the terminal run
`make noEntry` followed by
`./a.out 1` where 1 can be replaced by the number of the image file you wish to process.
