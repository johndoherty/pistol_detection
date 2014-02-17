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
                   std::vector<std::vector<Point> >& results, std::vector<float>& costs,
                   double templScale, int maxMatches, double minMatchDistance, int padX,
                   int padY, int scales, double minScale, double maxScale,
                   double orientationWeight, double truncate );
#endif
