//
//  chamfer.h
//  PistolDetection
//
//  Created by John Doherty on 2/13/14.
//  Copyright (c) 2014 John Doherty, Aaron Damashek. All rights reserved.
//

#ifndef PistolDetection_chamfer_h
#define PistolDetection_chamfer_h

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int chamerMatching( Mat& img, Mat& templ,
                   CV_OUT vector<vector<Point> >& results, CV_OUT vector<float>& cost,
                   double templScale=1, int maxMatches = 20,
                   double minMatchDistance = 1.0, int padX = 3,
                   int padY = 3, int scales = 5, double minScale = 0.6, double maxScale = 1.6,
                   double orientationWeight = 0.5, double truncate = 20);

#endif
