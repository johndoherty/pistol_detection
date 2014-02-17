//
//  main.cpp
//  PistolDetection
//
//  Created by John Doherty on 2/10/14.
//  Copyright (c) 2014 John Doherty, Aaron Damashek. All rights reserved.
//
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <iostream>

using namespace std;
using namespace cv;

//#include "chamfer.h"

vector<vector<int>> truth;
int numPolygons = 10;

bool basicChamfer(Mat img, Mat tpl){
    Canny(img, img, 100, 300, 3);
    Canny(tpl, tpl, 100, 300, 3);
    
    
    vector<vector<Point> > results;
    vector<float> costs;
    
    int best = chamerMatching(img, tpl, results, costs);
    if( best < 0 ) {
        cout << "not found;\n";
        return false;
    }
    return true;
}

/*To be implemented*/
Vector<Mat> splitIntoImages(Mat img){
    Vector<Mat> subImages;
    return subImages;
}

bool votingChamfer(Mat img, Mat tpl){
	Vector<Mat> subPolygons = splitIntoImages(img);
	int detected = 0;
	for(int i = 0; i < numPolygons; i++){
		if(basicChamfer(subPolygons[i], tpl)) detected += 1;
	}
	if(detected > numPolygons/2) return true;
	return false;
}

void setUpMLChamfer(Mat tpl, Vector<Mat> trainingImages){
	Vector<Vector<bool>> trainingSamples;
    vector<bool> found;
	for(int j = 0; j < trainingImages.size(); j++){
        Vector<Mat> subPolygons = splitIntoImages(trainingImages[j]);
        for(int i = 0; i < numPolygons; i++){
            if(basicChamfer(subPolygons[i], tpl)) found.push_back(true);
            else found.push_back(false);
        }
        trainingSamples.push_back(found);
	}
	//machine.train(trainingSamples);
	//return Machine;
}

bool MLChamfer(Mat img, Mat tpl){
//bool MLChamfer(Mat img, Mat tpl, machine){
	Vector<Mat> subPolygons = splitIntoImages(img);
	//if(!machine.trained) return;
	vector<bool> found;
	for(int i = 0; i < numPolygons; i++){
		if(basicChamfer(subPolygons[i], tpl)) found.push_back(true);
		else found.push_back(false);
	}
	//return machine(found);
    return false;
}

void testFunction(bool (*chamferFunction)(Mat img, Mat tpl), Mat tpl){
    int falsePositives = 0;
    int falseNegatives = 0;
    int correctIdentification = 0;
    int correctDiscard = 0;
    for(int i = 1; i <= 120; i++){//Denoting the imageNumber
        int imgNum = 1;
        while(true){
            string folder = to_string(i);
            string pic = to_string(imgNum);
            if(i < 10) folder = "0" + folder;
            if(imgNum < 10) pic = "0" + pic;
            Mat img = imread("../X0" + folder + "_" + pic + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
            Mat cimg;
            cvtColor(img, cimg, CV_GRAY2BGR);
            if(img.data) break;
            bool gunFound = chamferFunction(img, tpl); //Basic, votingChamfer or MLChamfer
            if(gunFound && truth[i][imgNum-1]){
                correctIdentification+=1;
            }else{
                falsePositives+=1;
            }
            if(!gunFound && !truth[i][imgNum-1]){
                correctDiscard+=1;
            }else{
                falseNegatives+=1;
            }
        }
    }
    int sum = falsePositives + falseNegatives + correctDiscard + correctIdentification;
    cout << "False positives: " << falsePositives << endl;
    cout << "False Negatives: " << falseNegatives << endl;
    cout << "Correct identifications: " << correctIdentification << endl;
    cout << "Correct Discards: " << correctDiscard << endl;
    cout << "Success rate: " << (double)(correctDiscard + correctIdentification)/sum*100;
}

int main( int argc, char** argv ) {
    
    if( argc != 1 && argc != 3 ) {
        return 0;
    }
    
    Mat img = imread(argc == 3 ? argv[1] : "./X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat cimg;
    cvtColor(img, cimg, CV_GRAY2BGR);
    Mat tpl = imread(argc == 3 ? argv[2] : "./X077_03.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    
    Canny(img, img, 100, 300, 3);
    Canny(tpl, tpl, 100, 300, 3);
    
    
    vector<vector<Point> > results;
    vector<float> costs;
    
    int best = chamerMatching(img, tpl, results, costs);
    if( best < 0 ) {
        cout << "not found;\n";
        return 0;
    }
    size_t j,m = results.size();
    for(j = 0; j < m; j++) {
        //size_t i, n = results[best].size();
        size_t i, n = results[j].size();
        for( i = 0; i < n; i++ ) {
            Point pt = results[best][i];
            if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
                cimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
        }
    }
    imshow("result", cimg);
    waitKey();
    return 0;
}

