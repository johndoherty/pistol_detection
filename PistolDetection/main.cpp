//
//  main.cpp
//  PistolDetection
//
//  Created by John Doherty on 2/10/14.
//  Copyright (c) 2014 John Doherty, Aaron Damashek. All rights reserved.
//
/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";


 //@function CannyThreshold
 //brief Trackbar callback - Canny thresholds input with a ratio 1:3

void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );
    
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}


int main( int argc, char** argv )
{
    /// Load an image
    src = imread("/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/X077_03.jpg");
    
    if( !src.data )
    { return -1; }
    
    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    
    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    
    /// Create a window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    
    /// Show the image
    CannyThreshold(0, 0);
    
    /// Wait until user exit program by pressing a key
    waitKey(0);
    
    return 0;
}
*/

#include "chamfer.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout <<
    "\nThis program demonstrates chamfer matching -- computing a distance between an \n"
    "edge template and a query edge image.\n"
    "Call:\n"
    "./chamfermatching [<image edge map> <template edge map>]\n"
    "By default\n"
    "the inputs are ./chamfermatching logo_in_clutter.png logo.png\n"<< endl;
}
/*int main( int argc, char** argv )
=======

int main( int argc, char** argv )
>>>>>>> 913e5df3af9fbd014e2bf225da24018b8a2ef41d
{
    if( argc != 1 && argc != 3 )
    {
        help();
        return 0;
    }
  */
/*
    Mat img = imread(argc == 3 ? argv[1] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo_in_clutter.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat tpl = imread(argc == 3 ? argv[2] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo.png", CV_LOAD_IMAGE_GRAYSCALE);
    */
    /*
    Mat img = imread(argc == 3 ? argv[1] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat tpl = imread(argc == 3 ? argv[2] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);*/
    
   // Mat img = imread(argc == 3 ? argv[1] : "/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/logo_in_clutter.png", CV_LOAD_IMAGE_GRAYSCALE);
//    Mat tpl = imread(argc == 3 ? argv[2] : "/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/logo.png", CV_LOAD_IMAGE_GRAYSCALE);

    //Mat img = imread(argc == 3 ? argv[1] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo_in_clutter.png", CV_LOAD_IMAGE_GRAYSCALE);
    //Mat tpl = imread(argc == 3 ? argv[2] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo.png", CV_LOAD_IMAGE_GRAYSCALE);

    //Mat img = imread(argc == 3 ? argv[1] : "./X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    //Mat tpl = imread(argc == 3 ? argv[2] : "./X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    // if the image and the template are not edge maps but normal grayscale images,
    // you might want to uncomment the lines below to produce the maps. You can also
    // run Sobel instead of Canny.

  //  Canny(img, img, 5, 50, 3);
   // Canny(tpl, tpl, 5, 50, 3);

    //imshow( "img", img );
    //imshow( "template", tpl );
    //waitKey(0);


    //Canny(img, img, 150, 200, 3);
    //Canny(tpl, tpl, 150, 200, 3);

/*
 char* window_name = "Edge Map";
 namedWindow( window_name, CV_WINDOW_AUTOSIZE );
 imshow(window_name, img);
*/
    //char * dir = getcwd(NULL, 0);
    //cout << "Current dir: " << dir << endl;


  //  vector<vector<Point> > results;
   // vector<float> costs;
//    int best = chamerMatching( img, tpl, results, costs );

  //  int best = chamerMatching(img, tpl, results, costs);
    //vector<vector<Point> > results;
    //vector<float> costs;
    //int best = chamerMatching(img, tpl, results, costs );
/*
    if( best < 0 )
    {
        cout << "not found;\n";
        return 0;
    }
    
    size_t i, n = results[best].size();
    for( i = 0; i < n; i++ )
    {
        Point pt = results[best][i];
        if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
            cimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
    }
    imshow("result", cimg);
*/
   // waitKey();
   // return 0;
//}

