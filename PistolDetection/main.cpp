//
//  main.cpp
//  PistolDetection
//
//  Created by John Doherty on 2/10/14.
//  Copyright (c) 2014 John Doherty, Aaron Damashek. All rights reserved.
//
//#include "chamfer.h"

#include "opencv2/contrib/contrib.hpp"
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

int main( int argc, char** argv ) {
    
    if( argc != 1 && argc != 3 ) {
        help();
        return 0;
    }
    
    /*
     Mat img = imread(argc == 3 ? argv[1] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo_in_clutter.png", CV_LOAD_IMAGE_GRAYSCALE);
     Mat tpl = imread(argc == 3 ? argv[2] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/logo.png", CV_LOAD_IMAGE_GRAYSCALE);
     */
    /*
     Mat img = imread(argc == 3 ? argv[1] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
     Mat tpl = imread(argc == 3 ? argv[2] : "/Users/aarondamashek/CS231A/pistol_detection/PistolDetection/X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);*/
    
    Mat img = imread(argc == 3 ? argv[1] : "/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/pistol_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat cimg;
    cvtColor(img, cimg, CV_GRAY2BGR);
    //Mat tpl = imread(argc == 3 ? argv[2] : "/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/logo.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat tpl = imread(argc == 3 ? argv[1] : "/Users/john/Dropbox/School/CS231a/Project/pistol_detection/PistolDetection/pistol_black_small.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
    
    // if the image and the template are not edge maps but normal grayscale images,
    // you might want to uncomment the lines below to produce the maps. You can also
    // run Sobel instead of Canny.
    
    Canny(img, img, 70, 300, 3);
    Canny(tpl, tpl, 150, 500, 3);

    imshow( "img", img );
    imshow( "template", tpl );
    waitKey(0);
    destroyAllWindows();
    
    imshow( "img", img );
    waitKey(0);
    destroyAllWindows();

    vector<vector<Point> > results;
    vector<float> costs;
    
    /*
     int chamerMatching( Mat& img, Mat& templ,
     CV_OUT vector<vector<Point> >& results, CV_OUT vector<float>& cost,
     double templScale=1, int maxMatches = 20,
     double minMatchDistance = 1.0, int padX = 3,
     int padY = 3, int scales = 5, double minScale = 0.6, double maxScale = 1.6,
     double orientationWeight = 0.5, double truncate = 20);
     */
    
    int best = chamerMatching(img, tpl, results, costs, 1, 50, 1.0, 3, 3, 5, 0.6, 1.6, 0.5, 20);
    //int best = chamerMatching(img, tpl, results, costs);
    if( best < 0 ) {
        cout << "not found;\n";
        return 0;
    }
    size_t j,m = results.size();
    //size_t j = best;
    for(j = 0; j < m; j++) {
        //size_t i, n = results[best].size();
        size_t i, n = results[j].size();
        for( i = 0; i < n; i++ ) {
            Point pt = results[j][i];
            if(pt.inside(Rect(0, 0, cimg.cols, cimg.rows))) {
                if (i == best) {
                    cimg.at<Vec3b>(pt) = Vec3b(255, 0, 0);
                } else {
                    cimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
                }
            }
            
        }
    }
    
    cout << "Best index: ";
    cout << best;
    cout << "\n";
    cout << "With cost: ";
    cout << costs[best];
    cout << "\n\n";
    
    for (int i = 0; i < costs.size(); i++) {
        cout << costs[i];
        cout << ", ";
    }
    imshow("result", cimg);
    imshow("edges", img);
    waitKey();
    return 0;
}
