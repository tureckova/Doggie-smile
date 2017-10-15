#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <Windows.h>
#include <numeric>
#include <time.h>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace dlib;

// define net type for dogHeadDetector

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// function to create vector of class names
std::vector<String> createClaseNames() {
	std::vector<String> classNames;
	classNames.push_back("background");
	classNames.push_back("aeroplane");
	classNames.push_back("bicycle");
	classNames.push_back("bird");
	classNames.push_back("boat");
	classNames.push_back("bottle");
	classNames.push_back("bus");
	classNames.push_back("car");
	classNames.push_back("cat");
	classNames.push_back("chair");
	classNames.push_back("cow");
	classNames.push_back("diningtable");
	classNames.push_back("dog");
	classNames.push_back("horse");
	classNames.push_back("motorbike");
	classNames.push_back("person");
	classNames.push_back("pottedplant");
	classNames.push_back("sheep");
	classNames.push_back("sofa");
	classNames.push_back("train");
	classNames.push_back("tvmonitor");
	return classNames;
}

//prepare the sounds options for attracting the dogs
std::vector<LPCWSTR> createSoundNames() {
	std::vector<LPCWSTR> sounds;
	sounds.push_back(L"cat.wav");
	sounds.push_back(L"cat2.wav");
	sounds.push_back(L"squeaky_toy.wav");
	sounds.push_back(L"one_squeak.wav");
	return sounds;
}

// function to play the sound
void playRndSound(std::vector<LPCWSTR> sounds)
{
	// choose rondomly the sound and play it
	std::srand(time(NULL));
	int s = std::rand() % sounds.size();
	PlaySound(sounds[s], NULL, SND_ASYNC);
}

// dog detector
int dogDetector(Mat &img, Size imgSize, Net net, std::vector<String> classNames) {

	// create input blob
	Mat img300;
	resize(img, img300, Size(300, 300));
	Mat inputBlob = blobFromImage(img300, 0.007843, Size(300, 300), Scalar(127.5)); //Convert Mat to dnn::Blob image batch

	// apply the blob on the input layer
	net.setInput(inputBlob); //set the network input

	// classify the image by applying the blob on the net
	Mat detections = net.forward("detection_out"); //compute output

	// look what the detector found
	int nrDog = 0;
	for (int i = 0; i < detections.size[2]; i++) {

		// detected class
		int indxCls[4] = { 0, 0, i, 1 };
		int cls = detections.at<float>(indxCls);

		// confidence
		int indxCnf[4] = { 0, 0, i, 2 };
		float cnf = detections.at<float>(indxCnf);

		// mark with bbox only dogs
		if (cls == 12 && cnf > 0.3) {
			// count the dog
			nrDog = nrDog++;
			// bounding box
			int indxBx[4] = { 0, 0, i, 3 };
			int indxBy[4] = { 0, 0, i, 4 };
			int indxBw[4] = { 0, 0, i, 5 };
			int indxBh[4] = { 0, 0, i, 6 };
			int Bx = detections.at<float>(indxBx) * imgSize.width;
			int By = detections.at<float>(indxBy) * imgSize.height;
			int Bw = detections.at<float>(indxBw) * imgSize.width - Bx;
			int Bh = detections.at<float>(indxBh) * imgSize.height - By;

			// draw bounding box to image
			Rect bbox(Bx, By, Bw, Bh);
			Scalar color(255, 0, 255);
			cv::rectangle(img, bbox, color, 1, 8, 0);
			String text = classNames[cls] + ", conf: " + to_string(cnf * 100);
			putText(img, text, Point(Bx, By), FONT_HERSHEY_SIMPLEX, 0.5, color);
		}

	}

	return nrDog;
}

// count euclidean distance between two points
float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

// Check the head position
bool checkHeadPosition(const dlib::full_object_detection& landmarks)
{
	// maximum difference in the distances
	float maxError = 0.3;

	//distance between left eye and nose and right eye and nose should be nearly the same
	float diffLeye = euclideanDist(cv::Point(landmarks.part(3).x(), landmarks.part(3).y()), cv::Point(landmarks.part(5).x(), landmarks.part(5).y()));
	float diffReye = euclideanDist(cv::Point(landmarks.part(3).x(), landmarks.part(3).y()), cv::Point(landmarks.part(2).x(), landmarks.part(2).y()));
	float meanEyeDist = (diffReye + diffLeye) / 2;
	// if the distance difference is bigger than 20% of its value return false
	if ((diffLeye - diffReye) > meanEyeDist * maxError || (diffLeye - diffReye) < -1 * meanEyeDist * maxError) {
		return false;
	}

	//distance between left ear and nose and right ear and nose should be nearly the same
	float diffLear = euclideanDist(cv::Point(landmarks.part(3).x(), landmarks.part(3).y()), cv::Point(landmarks.part(4).x(), landmarks.part(4).y()));
	float diffRear = euclideanDist(cv::Point(landmarks.part(3).x(), landmarks.part(3).y()), cv::Point(landmarks.part(1).x(), landmarks.part(1).y()));
	float meanEarDist = (diffLear + diffRear) / 2;
	// if the distance difference is bigger than 20% of its value return false
	if ((diffLear - diffRear) > meanEarDist * maxError || (diffLear - diffRear) < -1 * meanEarDist * maxError) {
		return false;
	}

	// if all distances are equal return true
	return true;

}

// Draw dog face
void renderFace(cv::Mat &img, const dlib::full_object_detection& landmarks, cv::Scalar color)
{
	// save the point into vector
	std::vector <cv::Point> points;
	points.push_back(cv::Point(landmarks.part(0).x(), landmarks.part(0).y())); //top
	points.push_back(cv::Point(landmarks.part(1).x(), landmarks.part(1).y())); //rear
	points.push_back(cv::Point(landmarks.part(2).x(), landmarks.part(2).y())); //reye
	points.push_back(cv::Point(landmarks.part(3).x(), landmarks.part(3).y())); //nose
	points.push_back(cv::Point(landmarks.part(5).x(), landmarks.part(5).y())); //leye
	points.push_back(cv::Point(landmarks.part(4).x(), landmarks.part(4).y())); //lear

	// draw head
	cv::polylines(img, points, true, color, 1, 16);
}

// find dog's head and check dog smile
int dogSmileDetector(Mat &img, net_type netHead, shape_predictor landmarkDetector) {

	// convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
	Mat imRGB;
	cv::cvtColor(img, imRGB, cv::COLOR_BGR2RGB);
	dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

	//detect dog heads in image
	std::vector<dlib::mmod_rect>faceRects = netHead(imDlib);

	// Loop over all detected face rectangles
	int headPositionOk = 0;
	for (int i = 0; i < faceRects.size(); i++)
	{
		// For every face rectangle, run landmarkDetector
		full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i].rect);

		// Check head position
		if (checkHeadPosition(landmarks)) {
			// Draw landmarks on face (green)
			renderFace(img, landmarks, cv::Scalar(0, 255, 0));
			// The head position is OK, so end the loop and continue
			headPositionOk = headPositionOk++;
		}
		else {
			// Draw landmarks on face (red)
			renderFace(img, landmarks, cv::Scalar(0, 0, 255));
		}
	}

	return headPositionOk;

}

// main function
int main(int argc, char **argv)
{
	// set inputs
	String pathNetDogTxt("MobileNetSSD_deploy.prototxt");
	String pathNetDogBin("MobileNetSSD_deploy.caffemodel");
	String pathNetHead("dogHeadDetector.dat");
	String pathLandmarkDetector("landmarkDetector.dat");
	std::vector<String> classNames = createClaseNames();
	std::vector<LPCWSTR> sounds = createSoundNames();

	//read all models
	Net netDog;
	net_type netHead;
	shape_predictor landmarkDetector;
	try {
		//read caffe model
		netDog = readNetFromCaffe(pathNetDogTxt, pathNetDogBin);
		// Load the dog head detector
		deserialize(pathNetHead) >> netHead;
		// Load landmark model
		deserialize(pathLandmarkDetector) >> landmarkDetector;
	}
	catch (cv::Exception& e) {
		cerr << "Exception: " << e.what() << std::endl;
		if (netDog.empty())
		{
			cerr << "Can't load caffe model." << std::endl;
			exit(-1);
		}
	}

	// Create a VideoCapture object
	cv::VideoCapture cap(0);
	// Check if OpenCV is able to read feed from camera
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		cv::waitKey(0);
	}
	// define Mat 
	Mat img;

	// grab first image and get its size
	cap >> img;
	Size imgSize = img.size();

	// Grab and process frames until the main window is closed by the user.
	bool state = true;
	int frameCount = 5;
	int dogInFrameCount = 0;
	int nrDogs, nrDogsPrev, headPositionOk;
	while (state) {

		// grab frame
		cap >> img;

		// if the enought time elapsed and number of dogs did not changed in last 5 frames play sound
		if (frameCount > 70 && dogInFrameCount >= 4) {
			// play sound
			playRndSound(sounds);
			// detect dogs in frame
			nrDogs = dogDetector(img, imgSize, netDog, classNames);
			// restart frame count
			frameCount = 0;
		}
		// exactly 4 frames after playing the sound, check that the number of dogs in frame did not changed and take picture and analyze
		if (frameCount == 4 && dogInFrameCount >= 4) {
			//create the copy of frame
			Mat imgCopy = img.clone();

			// detect dogs in frame
			nrDogs = dogDetector(img, imgSize, netDog, classNames);

			// detect dogs head and the landmarks in frame
			headPositionOk = dogSmileDetector(img, netHead, landmarkDetector);

			// create window with output image with displayed dog rectangle and landmarks
			String winName("Dog and Face detector output");
			imshow(winName, img);

			// print results
			cout << "---------------------" << endl;
			cout << "nrDogs: " << nrDogs << endl;
			cout << "headPositionOk: " << headPositionOk << endl;

			// check if is the number of dogs found and the number of head positions Ok the same
			if (nrDogs == headPositionOk) {

				// display taken picture
				String winName2("Taken foto OK");
				imshow(winName2, imgCopy);
				// save image
				String fileName("output.jpg");
				imwrite(fileName, imgCopy);
				// wait for keypress
				waitKey();
				// break loop
				state = false;
			}

			// restart frameCount
			frameCount = 5;
		}
		else {

			// detect dogs in frame
			nrDogs = dogDetector(img, imgSize, netDog, classNames);

		}

		// increase frameCount
		frameCount = frameCount++;

		// chceck if the number of dogs detected in last and current frame changed
		if (nrDogsPrev != nrDogs) {
			nrDogsPrev = nrDogs;
			dogInFrameCount = 0;
		}
		else {
			dogInFrameCount = dogInFrameCount++;
		}

		// show image
		String winName("image");
		imshow(winName, img);

		// Wait for keypress
		char key = cv::waitKey(1);
		if (key == 27) // ESC
		{
			// If ESC is pressed, exit.
			state = false;
		}

	}

	//destroy all windows and quit
	destroyAllWindows();

}


