//./facedetect --cascade="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml" --nested-cascade="/usr/share/opencv/haarcascades/haarcascade_eye.xml" --scale=1.3


// compile g++ -o facedetect facedetect.cpp `pkg-config opencv --cflags --libs opencv`  -lboost_system -lboost_filesystem -pthread `sdl-config --cflags --libs` -lSDL_mixer -std=c++11

#define SAYCOMPTEURLIMIT 20

#define LIMIT_IMAGE 20	
#define SEUILLE_Fisher 800
#define SEUILLE_LBPH 100
#define SEUILLE_LBPH_SUR 85
#define extraAcceptanceFisher 250


#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <pthread.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/face/facerec.hpp"

#include <stdio.h>
#include <stdlib.h>

#include "opencv2/videoio/videoio_c.h"
#include <glob.h>


#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <time.h>

#define BOOST_FILESYSTEM_DEPRECATED

#include <SDL/SDL.h>
#include <SDL/SDL_mixer.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"


using namespace std;
using namespace cv;
using namespace face;

namespace fs = boost::filesystem;

int numRand, numVoice, numPers;
bool saySomething = false;
int sayCompteur;
int numFaces;

vector<Mix_Music*> music;
vector<int> intLabels;
vector<Mat> images;
vector<string > nameLabels;
  
void antonyo(cv::Mat& img, Point& P, int& size, vector<string >& names);

vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

template <typename T>
std::string to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

int rows, cols;
  // Create a new Fisherfaces model and retain all available Fisherfaces,
  // this is the most common usage of this specific FaceRecognizer:
  //
 Ptr<FaceRecognizer> modelFisher =  createFisherFaceRecognizer();
 Ptr<FaceRecognizer> modelLBPH =  createLBPHFaceRecognizer();

//////////////////////////////////////////////////////////////
void uploadPerson( string& pathArg, vector<Mat>& imagesArg, vector<int >& intLabelsArg, string nameOfThisArg, int intOfThis)
{
	
  vector<boost::filesystem::path> photoFiles;
  fs::path full_path( fs::initial_path<fs::path>() );
  
  full_path = fs::system_complete( fs::path( pathArg ) );
  unsigned long file_count = 0;
  unsigned long dir_count = 0;
  unsigned long other_count = 0;
  unsigned long err_count = 0;

  if ( !fs::exists( full_path ) )
  {
    std::cout << "\nNot found: " << full_path.file_string() << std::endl;
  }

  if ( fs::is_directory( full_path ) )
  {
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator dir_itr( full_path );
          dir_itr != end_iter;
          ++dir_itr )
    {
      try
      {
        if ( fs::is_directory( dir_itr->status() ) )
        {
          ++dir_count;
        }
        else if ( fs::is_regular_file( dir_itr->status() ) )
        {
          ++file_count;
          
          if (file_count <= LIMIT_IMAGE) 
			{
				string fileName = dir_itr->path().filename().c_str();
				cout << pathArg + "/" + fileName << endl;
				
                imagesArg.push_back(imread(pathArg + "/" + fileName, CV_LOAD_IMAGE_GRAYSCALE));
                intLabelsArg.push_back( intOfThis );
            }
        }
        else
        {
          ++other_count;
        }

      }
      catch ( const std::exception & ex )
      {
        ++err_count;
        std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
      }
    }
  }	
	
}

//////////////////////////////////////////////////////////////
int uploadPhotos( string& pathArg, vector<Mat>& imagesArg, vector<string >& namesArg, vector<int >& intLabels )
{
  vector<boost::filesystem::path> photoFiles;
  fs::path full_path( fs::initial_path<fs::path>() );
  
  full_path = fs::system_complete( fs::path( pathArg ) );
  unsigned long file_count = 0;
  unsigned long dir_count = 0;
  unsigned long other_count = 0;
  unsigned long err_count = 0;

  if ( !fs::exists( full_path ) )
  {
    std::cout << "\nNot found: " << full_path.file_string() << std::endl;
    return -1;
  }

  if ( fs::is_directory( full_path ) )
  {
    std::cout << "\nIn directory: "
              << full_path.directory_string() << "\n\n";
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator dir_itr( full_path );
          dir_itr != end_iter;
          ++dir_itr )
    {
      try
      {
        if ( fs::is_directory( dir_itr->status() ) )
        {
          ++dir_count;
          std::cout << dir_itr->path().filename() << " [directory]\n";
          
          namesArg.push_back( dir_itr->path().filename().c_str() );
          
          string thisPersonPath = full_path.file_string() + dir_itr->path().filename().c_str();
          uploadPerson( thisPersonPath, imagesArg, intLabels, dir_itr->path().filename().c_str(), namesArg.size() - 1);
            
        }
        else if ( fs::is_regular_file( dir_itr->status() ) )
        {
        }
        else
        {
          ++other_count;
          std::cout << dir_itr->path().filename() << " [other]\n";
        }

      }
      catch ( const std::exception & ex )
      {
        ++err_count;
        std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
      }
    }
    std::cout << "\n" << file_count << " files\n"
              << dir_count << " directories\n"
              << other_count << " others\n"
              << err_count << " errors\n";
  }
  else // must be a file
  {
    std::cout << "\nFound: " << full_path.file_string() << "\n";    
  }
  
  return namesArg.size();
}
//////////////////////////////////////////////////////////////
int uploadMP3( string& pathArg, vector<Mix_Music*>& musicArg )
{
  vector<boost::filesystem::path> mp3files;
  boost::progress_timer t( std::clog );

  fs::path full_path( fs::initial_path<fs::path>() );
  
  full_path = fs::system_complete( fs::path( pathArg ) );

   if (SDL_Init(SDL_INIT_AUDIO) != 0)
   {
      std::cerr << "SDL_Init ERROR: " << SDL_GetError() << std::endl;
      return -1;
   }

   // Open Audio device
   if (Mix_OpenAudio(44100, AUDIO_S16SYS, 2, 2048) != 0)
   {
      std::cerr << "Mix_OpenAudio ERROR: " << Mix_GetError() << std::endl;
      return -1;
   }

   // Set Volume
   Mix_VolumeMusic(100);
   
  unsigned long file_count = 0;
  unsigned long dir_count = 0;
  unsigned long other_count = 0;
  unsigned long err_count = 0;

  if ( !fs::exists( full_path ) )
  {
    std::cout << "\nNot found: " << full_path.file_string() << std::endl;
    return -1;
  }

  if ( fs::is_directory( full_path ) )
  {
    std::cout << "\nIn directory: "
              << full_path.directory_string() << "\n\n";
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator dir_itr( full_path );
          dir_itr != end_iter;
          ++dir_itr )
    {
      try
      {
        if ( fs::is_directory( dir_itr->status() ) )
        {
          ++dir_count;
          std::cout << dir_itr->path().filename() << " [directory]\n";
        }
        else if ( fs::is_regular_file( dir_itr->status() ) )
        {
          ++file_count;
          string fileName = dir_itr->path().filename().c_str();
          int l=fileName.size();
          if ( (fileName[l-4] == '.') && (fileName[l-3] == 'm') && (fileName[l-2] == 'p') && (fileName[l-1] == '3') ) {
			mp3files.push_back(full_path.file_string() + dir_itr->path().filename().string());
			musicArg.push_back( Mix_LoadMUS(  (mp3files.back()).c_str() ) );
			cout << dir_itr->path().filename().string() << endl;
	   	}
        }
        else
        {
          ++other_count;
          std::cout << dir_itr->path().filename() << " [other]\n";
        }

      }
      catch ( const std::exception & ex )
      {
        ++err_count;
        std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
      }
    }
    std::cout << "\n" << file_count << " files\n"
              << dir_count << " directories\n"
              << other_count << " others\n"
              << err_count << " errors\n";
  }
  else // must be a file
  {
    std::cout << "\nFound: " << full_path.file_string() << "\n";    
  }
  
  return music.size();
}	  

//////////////////////////////////////////////////////////////
void play (int num, vector<Mix_Music*>& musicArg) {
   if (musicArg.size()>0)
   {
      // Start Playback
      if (Mix_PlayMusic(musicArg[num], 1) == 0)
      {
//         unsigned int startTime = SDL_GetTicks();

         // Wait
      //   while (Mix_PlayingMusic())
      //   {
      //      nanosleep((const struct timespec[]){{1, 0}}, NULL);
      //      std::cout << "Time: " << (SDL_GetTicks() - startTime) / 1000 << std::endl;
      //   }
      }
      else
      {
         std::cerr << "Mix_PlayMusic ERROR: " << Mix_GetError() << std::endl;
      }
   }
   else
   {
      std::cerr << "Mix_LoadMuS ERROR: " << Mix_GetError() << std::endl;
   }
}

//////////////////////////////////////////////////////////////
void exit (vector<Mix_Music*>& musicArg) {
   // Free File
   for (unsigned int i = 0 ; i < musicArg.size() ; ++i)
	Mix_FreeMusic(musicArg[i]);

   // End
   Mix_CloseAudio();
}

//////////////////////////////////////////////////////////////
static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

//////////////////////////////////////////////////////////////
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    int i = 0;
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    // printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center, centerToSave;
        Scalar color = colors[i%8];
        int radius, radiusToSave;
        numFaces = 0;
		
        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            //circle( img, center, radius, color, 3, 8, 0 );
            centerToSave = center ;
            radiusToSave = radius ;
            antonyo(img, centerToSave, radiusToSave, nameLabels);
        }
        else
            rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            //circle( img, center, radius, 10 /* color */, 3, 8, 0 );
        }
        
        // if (c >= 3) {
			cout << faces.size() << " faces." << endl;
			numFaces = faces.size();
			// antonyo(img, centerToSave, radiusToSave, nameLabels);
		// }
    }
    
    // cout << faces.size() << " face(s)" << endl;
    cv::imshow( "result", img );

}

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName = "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

// Will be used in functon antonyo
Mat img2 = imread("orl_faces/s1/5.pgm", CV_LOAD_IMAGE_GRAYSCALE);
    
//////////////////////////////////////////////////////////////
int main( int argc, const char** argv )
{
	string path = "audio/";
	string pathPhotos = "people/";
	
	numVoice = uploadMP3( path, music);
	numPers = uploadPhotos( pathPhotos, images, nameLabels, intLabels);

	cout << "images stores " << int(images.size()) << " images.\n";
	cout << "labels stores " << int(intLabels.size()) << " labels.\n";
	cout << "for " << int(nameLabels.size()) << " people.\n";

	  // This is the common interface to train all of the available cv::FaceRecognizer
	  // implementations:
	  //
	modelFisher->train(images, intLabels);
	modelLBPH->train(images, intLabels);

//	modelFisher->save("DatabaseFisher");
//	modelLBPH->save("DatabaseLBPH");

	cout << "Model Traines\n"; 
	  
	  		
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string nestedCascadeOpt = "--nested-cascade";
    size_t nestedCascadeOptLen = nestedCascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
    bool tryflip = false;

    help();

    CascadeClassifier cascade, nestedCascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
        {
            if( argv[i][nestedCascadeOpt.length()] == '=' )
                nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
            if( !nestedCascade.load( nestedCascadeName ) )
                cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
        {
            tryflip = true;
            cout << " will try to flip image horizontally to detect assymetric objects\n";
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }

    // cvNamedWindow( "result", 1 );

    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
			saySomething = false;
            IplImage* iplImg = cvQueryFrame( capture );
            frame = cv::cvarrToMat(iplImg);
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

			// cv::imshow( "result", frame );
            detectAndDraw( frameCopy, cascade, nestedCascade, scale, tryflip );
			if ( (saySomething == true) && (sayCompteur > SAYCOMPTEURLIMIT ) ) {
				srand(time(NULL));
				numRand = rand() % numVoice;
				
				if ( (numFaces) > 1 )
				  numRand = 1;
				else 
				  numRand = 3;
				  
				cout << "Random Number is : " << numRand << ". Face number " << numFaces << endl;

				play(numRand, music);
				sayCompteur = 0;
			}
			sayCompteur ++ ;
            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}

int antInt=0;
float multi;

void antonyo(cv::Mat& img, Point& P, int& size, vector<string >& names) {
   // The limits of the image
    multi = 1;
    saySomething = true;
    circle( img, P, size, 255 /*color*/, 3, 8, 0 );
    cv::Mat subImg = img(cv::Range(P.y-size*multi, P.y+size*multi), cv::Range(P.x-size*multi, P.x+size*multi));

// ference pour la taille
    cvtColor(subImg, subImg, CV_RGB2GRAY);        
    resize(subImg, subImg, img2.size());
  
    subImg.convertTo(subImg, -1, 1, 75);
    // cv::imshow( "subImg", subImg );
/*
    string intStr = to_string(antInt);
    string str = string(intStr);
    str = "antonyoNew/" + str + ".jpg";
    imwrite( str, subImg );
    antInt++;
*/
    
    
    // Some variables for the predicted label and associated confidence (e.g. distance):
  int predicted_labelLBPH = -1, predicted_labelFisher = -1, predicted_labelEigen = -1;
  double predicted_confidenceLBPH = 0.0, predicted_confidenceFisher = 0.0, predicted_confidenceEigen = 0.0;

  // Get the prediction and associated confidence from the model
  modelFisher->predict(subImg, predicted_labelFisher, predicted_confidenceFisher);
  modelLBPH->predict(subImg, predicted_labelLBPH, predicted_confidenceLBPH);

  cout << "Fisher : " << names[predicted_labelFisher] << " (" << predicted_confidenceFisher << " ). LBPH :" << names[predicted_labelLBPH] << " (" << predicted_confidenceLBPH << " )." << endl;

  string toWrite = "Inconnu";
  int color = 100;
  int acceptFisher=0;

  if (predicted_labelLBPH == predicted_labelFisher) {
	acceptFisher = extraAcceptanceFisher;
  }
  if (predicted_confidenceFisher < (SEUILLE_Fisher+acceptFisher)) {
	if (predicted_confidenceLBPH < SEUILLE_LBPH) {
		toWrite = names[predicted_labelLBPH];
		color = 255;
		if (predicted_confidenceLBPH < SEUILLE_LBPH_SUR) {
			vector<Mat> newImage;
			newImage.push_back(subImg);
	  		
			vector<int> newLabels;
			newLabels.push_back(predicted_labelLBPH);
			modelLBPH->update(newImage,newLabels);
		}
	}
  }
  /*
  if (predicted_confidenceFisher < SEUILLE_Fisher) {
	if (predicted_confidenceLBPH < SEUILLE_LBPH) {
		toWrite = names[predicted_labelLBPH];
		color = 255;	
	  	if (predicted_labelLBPH == predicted_labelFisher) {
	
			vector<Mat> newImage;
			newImage.push_back(subImg);
	  		
			vector<int> newLabels;
			newLabels.push_back(predicted_labelLBPH);
			modelLBPH->update(newImage,newLabels);
		}
	}
  }*/
	
  cv::putText(img, toWrite, P, 2, 2, Scalar::all(color), 2,8);


}
