/**
 * @file test_node.cpp
 * @author Bin Li
 * @brief 读取给定的视频，发送到识别节点进行一系列的测试
 * @date 2023-10-07
 * @version 0.2
 * @copyright Copyright (c) 2023 by Bin Li
 */

 #include "test/test_node.hpp"
 #include <fstream>
 #include <yaml-cpp/yaml.h>

 namespace rune
 {
 struct TestNode::Impl_
 {
   const std::string kTestVideo       = "7_28_test.mp4";
   const std::string kTestCamInfo     = "DA0710001.yaml";
   const std::string kPubImgTopicName = "/camera/front/capture";
   const std::string kPubCamTopicName = "/camera/front/camera_info";
   const int kTransSpeed              = 1000 / 30; // 即30帧/秒

 #if USE_ABS_PATH
   const std::string kTestVideoPath = "/home/cym/rune_video_and_data/" + kTestVideo;
   const std::string kTestCamPath = "/home/cym/AutoAim/src/ros_camera/camera_matrix/hik/" + kTestCamInfo;
 #else
   const std::string kPkgName         = "rune";
   const std::string kPkgShareDirPath = ament_index_cpp::get_package_share_directory(kPkgName);
   const std::string kTestVideoPath   = kPkgShareDirPath + "/video/" + kTestVideo;
 #endif

   cv::VideoCapture video_cap_;
   sensor_msgs::msg::CameraInfo cam_info_msg_;
   rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_puber_;
   rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_puber_;
   rclcpp::TimerBase::SharedPtr timer_;
 };
 
 TestNode::TestNode(const rclcpp::NodeOptions & options) : Node("TestNode", options), impl_(std::make_unique<Impl_>())
 {
   RCLCPP_INFO(this->get_logger(), "Starting TestNode ...");
 
   // 创建 ROS2 图像发布者
   impl_->img_puber_ = this->create_publisher<sensor_msgs::msg::Image>(impl_->kPubImgTopicName, 10);
   impl_->cam_puber_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(impl_->kPubCamTopicName, 10);
 
   // 打开视频文件
   impl_->video_cap_.open(impl_->kTestVideoPath);
   if (!impl_->video_cap_.isOpened()) {
     RCLCPP_FATAL(this->get_logger(), "Failed to open test-video: %s", impl_->kTestVideoPath.c_str());
     rclcpp::shutdown();
     return;
   }

   //打开相机参数文件
   std::ifstream cam_file(impl_->kTestCamPath);
   if(!cam_file.is_open()) {
    RCLCPP_FATAL(this->get_logger(),"Failed to open camera info file: %s", impl_->kTestCamPath.c_str());
    rclcpp::shutdown();
    return;
   }
 
   try {
    YAML::Node cam_params = YAML::LoadFile(impl_->kTestCamPath);
    impl_->cam_info_msg_.k = {
      cam_params["camera_matrix"]["data"][0].as<double>(),
       cam_params["camera_matrix"]["data"][1].as<double>(),
       cam_params["camera_matrix"]["data"][2].as<double>(),
       cam_params["camera_matrix"]["data"][3].as<double>(),
       cam_params["camera_matrix"]["data"][4].as<double>(),
       cam_params["camera_matrix"]["data"][5].as<double>(),
       cam_params["camera_matrix"]["data"][6].as<double>(),
       cam_params["camera_matrix"]["data"][7].as<double>(),
       cam_params["camera_matrix"]["data"][8].as<double>()
    };
    impl_->cam_info_msg_.d = cam_params["distortion_coefficients"]["data"].as<std::vector<double>>();
    impl_->cam_info_msg_.width = cam_params["image_width"].as<int>();
    impl_->cam_info_msg_.height = cam_params["image_height"].as<int>();
   }catch (const YAML::Exception &e) {
    RCLCPP_FATAL(this->get_logger(), "Failed to parse camera info file: %s", e.what());
    rclcpp::shutdown();
    return;
   }

   // 创建定时器，每隔 kTransSpeed 毫秒发布一帧图像与相机信息
   impl_->timer_ = this->create_wall_timer(
     std::chrono::milliseconds(impl_->kTransSpeed),
     std::bind(&TestNode::PublishFrameAndCam, this));
 
   RCLCPP_INFO(this->get_logger(), "TestNode initialized successfully.");
 }
 
 TestNode::~TestNode()
 {
   RCLCPP_INFO(this->get_logger(), "Ending TestNode ...");
   impl_->video_cap_.release();
 }
 
 // 逐帧发布视频图像与相机信息
 void TestNode::PublishFrameAndCam()
 {
   cv::Mat frame;
   if (!impl_->video_cap_.read(frame)) {
     RCLCPP_WARN(this->get_logger(), "Video playback finished, restarting...");
     impl_->video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
     return;
   }

   // 转换为 ROS2 图像消息并发布
   auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
   impl_->img_puber_->publish(*msg);

   // 发布相机信息
   impl_->cam_info_msg_.header.stamp = this->now();
   impl_->cam_puber_->publish(impl_->cam_info_msg_);

   RCLCPP_DEBUG(this->get_logger(), "Published a video frame and camera info.");
 }
 }  // namespace rune
 
 #include "rclcpp_components/register_node_macro.hpp"
 RCLCPP_COMPONENTS_REGISTER_NODE(rune::TestNode)
 