// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

// std includes
#include <iostream>
#include <string>
#include <regex>
#include <experimental/filesystem>

#define VERBOSE false

using namespace std;

void fetch_test_image_paths(std::string path, std::vector<std::string> &paths)
{
    std::cout << "Scanning dataset path \"" << path << "\"..." << std::endl;

    // Generate vector of paths to supported files
    paths.clear();
    std::vector<std::string> supported_file_extensions = {"jpg", "png"};
    for (const auto & entry : std::experimental::filesystem::directory_iterator(path))
    {
        std::string path_string = entry.path();
        std::string file_extension = path_string.substr(path_string.find_last_of(".") + 1);

        for(const auto & supported_file_extension : supported_file_extensions)
        {
            if(file_extension == supported_file_extension)
            {
                paths.push_back(path_string);
                break;
            }
        }
        std::sort(paths.begin(), paths.end());
    }
    std::cout << "Found " << paths.size() << " image(s)!" << std::endl;
}

float blur(std::string path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR );
    if( image.empty() )
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //https://stackoverflow.com/questions/24080123/opencv-with-laplacian-formula-to-detect-image-is-blur-or-not-in-ios
    cv::Mat matImageGrey;
    cv::cvtColor(image, matImageGrey, CV_BGR2GRAY);

    cv::Mat laplacianImage;
    //cv::Laplacian(matImageGrey, laplacianImage, CV_8U);

    //cv::Mat laplacianImage8bit;
    //laplacianImage.convertTo(laplacianImage8bit, CV_8UC1);

    cv::Laplacian(matImageGrey, laplacianImage, CV_64F);
    cv::Scalar mean, stddev; // 0:1st channel, 1:2nd channel and 2:3rd channel
    cv::meanStdDev(laplacianImage, mean, stddev, cv::Mat());
    double variance = stddev.val[0] * stddev.val[0];

    if(VERBOSE)
        std::cout << "Blur:\t\t" << variance << std::endl;

    return variance;
}

float occupancy(std::string path, int threshold)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR );
    if( image.empty() )
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::cvtColor( image, image, cv::COLOR_BGR2GRAY );
    //cv::equalizeHist( image, image );

    int pixels = 0;
    for(size_t row = 0; row < image.rows; row++)
    {
        for(size_t col = 0; col < image.cols; col++)
        {
            uchar intensity = image.at<uchar>(row,col);
            if(intensity > threshold)
                pixels++;
        }
    }

    float pct = (pixels*100.0)/(image.cols*image.rows);

    if(VERBOSE)
        std::cout << "occupancy (%):\t" << pct << std::endl;

    return pct;
}

float lightness(std::string path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR );
    if( image.empty() )
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::cvtColor( image, image, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( image, image );

    int pixels = 0;
    int64_t sum = 0;
    for(size_t row = 0; row < image.rows; row++)
    {
        for(size_t col = 0; col < image.cols; col++)
        {
            uchar intensity = image.at<uchar>(row,col);
            sum += intensity;
        }
    }

    float avg = (sum*1.0)/(image.cols*image.rows);

    if(VERBOSE)
        std::cout << "Lightness:\t\t" << avg << std::endl;

    return avg;
}

float similarity(std::string path1, std::string path2)
{
    cv::Mat image1 = cv::imread(path1, cv::IMREAD_COLOR );
    cv::Mat image2 = cv::imread(path2, cv::IMREAD_COLOR );
    if( image1.empty() || image2.empty())
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::cvtColor( image1, image1, cv::COLOR_BGR2GRAY );
    cv::cvtColor( image2, image2, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( image1, image1 );
    cv::equalizeHist( image2, image2 );

    int sum = 0;
    for(size_t row = 0; row < image1.rows; row++)
    {
        for(size_t col = 0; col < image1.cols; col++)
        {
            uchar intensity1 = image1.at<uchar>(row,col);
            uchar intensity2 = image2.at<uchar>(row,col);

            sum += std::abs(intensity2-intensity1);
        }
    }
    float pct = 100-((sum*100.0)/(255*image1.rows*image1.cols));

    if(VERBOSE)
        std::cout << "Similarity (%):\t" << pct << std::endl;

    return pct;
}

float similarity_hist(std::string path1, std::string path2)
{
    cv::Mat image1 = cv::imread(path1, cv::IMREAD_COLOR );
    cv::Mat image2 = cv::imread(path2, cv::IMREAD_COLOR );
    if( image1.empty() || image2.empty())
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::cvtColor( image1, image1, cv::COLOR_BGR2HSV );
    cv::cvtColor( image2, image2, cv::COLOR_BGR2HSV );
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };

    cv::Mat hist_image1, hist_image2;
    cv::calcHist( &image1, 1, channels, cv::Mat(), hist_image1, 2, histSize, ranges, true, false );
    cv::normalize( hist_image1, hist_image1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::calcHist( &image2, 1, channels, cv::Mat(), hist_image2, 2, histSize, ranges, true, false );
    cv::normalize( hist_image2, hist_image2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    double compare = cv::compareHist( hist_image1, hist_image2, 1 );

    //std::cout << compare << std::endl;
    return compare;
}

void process_images(const std::vector<std::string> &paths)
{
    std::string previous_path = paths[1];
    std::string best_path = "";
    float best_god_metric = 0;
    for(const string path : paths)
    {
        if(VERBOSE)
            std::cout << path << std::endl;
        float god_metric = 5*occupancy(path,0)+/*5*lightness(path)+*/4*similarity(previous_path,path);
        //std::cout << 5*occupancy(path,0) << "+" << 5*lightness(path) << "+" << 2*similarity(previous_path,path) << std::endl;
        //float god_metric = occupancy(path,0)+;
        //float god_metric = similarity_hist(previous_path,path) + 0.5*occupancy(path,0);

        //std::cout << "God metric:\t\t" << god_metric << std::endl;

        if(god_metric > best_god_metric)
        {
            best_god_metric = god_metric;
            best_path = path;
        }

        previous_path = path;
    }

    std::cout << "Best image: " << best_path.substr(best_path.find_last_of("/")+1, best_path.size()) << std::endl;
}

void get_data(std::vector<int> &image_data, std::string image_name, std::vector<std::string> delimiters = {"_i", "_s", "_z", "_e", ".jpg"})
{
    for(int j = 0; j < delimiters.size()-1; j++)
    {
        std::regex base_regex(delimiters[j] + "(.*)" + delimiters[j+1]);
        std::smatch base_match;
        std::regex_search(image_name, base_match, base_regex);
        image_data.push_back(std::stoi(base_match[1].str()));
    }
}

int main()
{
    cout << "[Image sequence analysis]" << endl;

    std::vector<std::string> delimiters = {"_s", "_i", "_z", ".jpg"};

    std::vector<std::string> paths;
    fetch_test_image_paths("/home/mal/MEGAsync/Images/garment-sequence-light-e25", paths);

    size_t index = 0;
    size_t sequence_index = 1;
    std::vector<std::string> sub_paths;

    while (true)
    {
        std::vector<int> image_data;
        get_data(image_data, paths[index], delimiters);
        if(image_data[0] == sequence_index && index != paths.size()-1)
        {
            sub_paths.push_back(paths[index]);
            index++;
        }
        else
        {
            std::cout << "[Sequence " << sequence_index << "]: ";
            process_images(sub_paths);
            sub_paths.clear();
            sequence_index++;

            if(index == paths.size()-1)
                break;
        }
    }

    /*

    cv::namedWindow("Fastwindow", CV_WINDOW_AUTOSIZE);
    cv::Mat image = cv::imread("/home/mal/MEGAsync/Images/garment-sequence-dark-e25/export_s0019_i0019_z1167.jpg", cv::IMREAD_COLOR );
    if( image.empty() )
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::cvtColor( image, image, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( image, image );
    cv::imshow("Fastwindow", image);
    cv::waitKey(0);
    */



    return 0;
}
