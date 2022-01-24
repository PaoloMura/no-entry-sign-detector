face: face.cpp
	g++ face.cpp /usr/lib64/libopencv_core.so.2.4 \
	/usr/lib64/libopencv_highgui.so.2.4 \
	/usr/lib64/libopencv_imgproc.so.2.4 \
	/usr/lib64/libopencv_objdetect.so.2.4;

# noEntry: noEntry.cpp hough.cpp hough.h
# 	g++ -std=c++11 noEntry.cpp hough.cpp /usr/lib64/libopencv_core.so.2.4 \
# 	/usr/lib64/libopencv_highgui.so.2.4 \
# 	/usr/lib64/libopencv_imgproc.so.2.4 \
# 	/usr/lib64/libopencv_objdetect.so.2.4

# hough: hough.cpp
# 	g++ -std=c++11 hough.cpp \
# 	/usr/lib64/libopencv_core.so.2.4 \
# 	/usr/lib64/libopencv_highgui.so.2.4 \
# 	/usr/lib64/libopencv_imgproc.so.2.4 \
# 	-o hough

noEntry: noEntry.cpp houghOOP.cpp houghOOP.h
	g++ -std=c++11 noEntry.cpp houghOOP.cpp \
	/usr/lib64/libopencv_core.so.2.4 \
	/usr/lib64/libopencv_highgui.so.2.4 \
	/usr/lib64/libopencv_imgproc.so.2.4 \
	/usr/lib64/libopencv_objdetect.so.2.4
