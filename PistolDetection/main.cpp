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


vector<vector<int>> truth;
int numPolygons = 10;
double matchThreshold = .15;

void populateTruth(){
    vector<int> firstFolder;
    for(int i = 0; i < 14; i++){
        firstFolder.push_back(0);
    }
    truth.push_back(firstFolder);
}

bool basicChamfer(Mat img, Mat tpl){
    Canny(img, img, 100, 300, 3);
    Canny(tpl, tpl, 100, 300, 3);
    
    
    vector<vector<Point> > results;
    vector<float> costs;
    
    int best = chamerMatching(img, tpl, results, costs);
    if( best < 0 || costs[best] < matchThreshold) {
        //cout << "not found;\n";
        return false;
    }
    return true;
}

/*To be implemented*/
vector<Mat> splitIntoImages(Mat img){
    vector<Mat> subImages;
    return subImages;
}

bool votingChamfer(Mat img, Mat tpl){
	vector<Mat> subPolygons = splitIntoImages(img);
	int detected = 0;
	for(int i = 0; i < numPolygons; i++){
		if(basicChamfer(subPolygons[i], tpl)) detected += 1;
	}
	if(detected > numPolygons/2) return true;
	return false;
}

void setUpMLChamfer(Mat tpl, vector<Mat> trainingImages){
	vector<vector<bool>> trainingSamples;
    vector<bool> found;
	for(int j = 0; j < trainingImages.size(); j++){
        vector<Mat> subPolygons = splitIntoImages(trainingImages[j]);
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
	vector<Mat> subPolygons = splitIntoImages(img);
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
    for(int i = 1; i <= 1; i++){//Denoting the folder
        int imgNum = 1;
        while(true){
            string folder = to_string(i);
            string pic = to_string(imgNum);
            if(i < 10) folder = "0" + folder;
            if(imgNum < 10) pic = "0" + pic;
            string fileLocation = "../../images/X0" + folder + "/X0" + folder + "_" + pic + ".png";
            Mat img = imread(fileLocation, CV_LOAD_IMAGE_GRAYSCALE);
            Mat cimg;
            cvtColor(img, cimg, CV_GRAY2BGR);
            if(!img.data) break;
            bool gunFound = chamferFunction(img, tpl); //Basic, votingChamfer or MLChamfer
            if(gunFound){
                if(truth[i-1][imgNum-1]){
                correctIdentification+=1;
                }else{
                    falsePositives+=1;
                }
            }
            if(!gunFound){
                if(!truth[i-1][imgNum-1]){
                correctDiscard+=1;
                }else{
                    falseNegatives+=1;
                }
            }
            imgNum++;
        }
    }
    int sum = falsePositives + falseNegatives + correctDiscard + correctIdentification;
    cout << "False positives: " << falsePositives << endl;
    cout << "False Negatives: " << falseNegatives << endl;
    cout << "Correct identifications: " << correctIdentification << endl;
    cout << "Correct Discards: " << correctDiscard << endl;
    cout << "Success rate: " << (double)(correctDiscard + correctIdentification)/sum*100 << endl;
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

