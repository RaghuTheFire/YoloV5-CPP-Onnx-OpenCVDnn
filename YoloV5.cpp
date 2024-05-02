// Include Libraries
#include <fstream>
#include <opencv2/opencv.hpp>

// Namespaces
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Text parameters
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Loading label names
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    //std::ifstream ifs("coco.names");
    std::ifstream ifs("classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

// Loading Onnx format weight file
void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("yolov5n.onnx");
    
    // Using CPU or GPU based on available system
    if (is_cuda)
    {
        std::cout << "Running on GPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

// Colors
const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

// Constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) 
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) 
{
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    
    // Convert to blob
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    // Forward propagate
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Resizing factor
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    // Initialize vectors to hold respective outputs while unwrapping detections    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Iterate through 25200 detections
    for (int i = 0; i < rows; ++i) 
    {

        float confidence = data[4];

        // Discard bad detections and continue
        if (confidence >= CONFIDENCE_THRESHOLD) 
        {

            float * classes_scores = data + 5;

            // Create a 1x85 Mat and store class scores of 'n' no.of classes
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;

            // Perform minMaxLoc and acquire index of best class score
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // Continue if the class score is above the threshold
            if (max_class_score > SCORE_THRESHOLD) 
            {

                // Store class ID and confidence in the pre-defined respective vectors
                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                // Center
                float x = data[0];
                float y = data[1];

                // Box dimension
                float w = data[2];
                float h = data[3];

                // Bounding box coordinates
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                // Store good detections in the boxes vector
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jumping to the next column
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) 
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        // cout<<result.class_id<<endl;
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{

    std::vector<std::string> class_list = load_class_list();

    //Load input video/image 
    cv::Mat frame;
    cv::VideoCapture capture("/home/bstc/Desktop/yolov5-opencv-cpp-python/test.mp4",cv::CAP_FFMPEG);
    //cv::VideoCapture capture(0);
    if (!capture.isOpened())
    {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    cv::dnn::Net net;
    load_net(net, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    //cv::VideoWriter::VideoWriter("out.avi",cv::Size(INPUT_WIDTH, INPUT_HEIGHT),true);
    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            // cout << detection.class_id <<endl;

            // Draw bounding box
            // cv::rectangle(frame, box, color, 3);
            // cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y,), color, cv::FILLED);
            cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y +box.height), color, 0);
            // --------------------
            std::ostringstream fps_label;
            fps_label << "FPS: " << 500;
            std::string fps_label_str = fps_label.str();
            cv::putText(frame, fps_label_str.c_str(), cv::Point(500, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);            
            // --------------------
            // Draw class labels
            cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        if (frame_count >= 10)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();
            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);
        }

        cv::imshow("output", frame);
        cv::waitKey(1);
        //Initialize video writer object
        //cv::VideoWriter output("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),frame);
        if (cv::waitKey(1) != -1)
        {
            capture.release();
            std::cout << "Done!!\n";
            break;
        }
    }

    std::cout << "Total frames: " << total_frames << "\n";
    return 0;
}
