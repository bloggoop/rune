#include "OpenVINO/OpenVINO_node.hpp"

namespace rune 
{
OpenVINO_node::OpenVINO_node(const rclcpp::NodeOptions &options) :
rclcpp::Node("OpenVINO_node", options)
{
  RCLCPP_INFO(this->get_logger(), "OpenVINO_node start ...");
  declareParams();
  RCLCPP_INFO(this->get_logger(), "declareParams finished");

  //创建推理核与R识别
  //TODO:加入参数中
  openvino_engine_target_ = std::make_unique<OpenVINO_rune>(target_model_path_, cv::Size(640,640), 0.8, 0.8);
  if (!openvino_engine_target_)
  {
    RCLCPP_ERROR(get_logger(), "openvino_engine_target_ is empty ptr");
  }
  ros_sub_channels_ -> camera_info_sub_ = this -> create_subscription<sensor_msgs::msg::CameraInfo>(
    ros_sub_topic_->camera_info_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&OpenVINO_node::CameraInfoCallback, this, std::placeholders::_1)
  );
  ros_sub_channels_-> img_sub_ = this -> create_subscription<sensor_msgs::msg::Image>(
    ros_sub_topic_->raw_img_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&OpenVINO_node::ImgCallback, this, std::placeholders::_1)
  );

  ros_sub_channels_-> pre_ready_sub_ = this -> create_subscription<std_msgs::msg::Bool>(
    "/rune/pre_ready",
    rclcpp::SensorDataQoS(),
    std::bind(&OpenVINO_node::preReadyCallback, this, std::placeholders::_1)
  );

  RCLCPP_INFO(this->get_logger(), "create_subscription finished");

  ros_pub_channels_ -> fanblade_pub_ = this -> create_publisher<rune_sys_interfaces::msg::Fanblade>(
    ros_pub_topic_ -> rec_info_topic_, 10);

  using std::chrono_literals::operator""ms;
  status_timer_ = 
    this->create_wall_timer(200ms, std::bind(&OpenVINO_node::updateNodeStatus, this));
  const int node_status = this->declare_parameter<int>("NodeStatus", 1);
  node_status_ = (node_status == 1) ? NodeStatus::ON_ACTIVATE : NodeStatus::ON_DEACTIVATE;

  if(debug_) {
    RCLCPP_WARN(this-> get_logger(), "Rune-Debug Mode start, Initializing debug-pub channels");
    ros_pub_channels_ ->debug_dif_angle_ =
      this->create_publisher<std_msgs::msg::Float64>(ros_pub_topic_ -> debug_dif_angle_topic_, rclcpp::SensorDataQoS());
    ros_pub_channels_ ->debug_raw_img_pub_ = 
      image_transport::create_publisher(this, ros_pub_topic_ ->raw_img_topic);
    ros_pub_channels_ ->debug_rec_img_pub_ = 
      image_transport::create_publisher(this, ros_pub_topic_ ->rec_img_topic_);
    ros_pub_channels_->debug_pnp_x_pub_ =
      this->create_publisher<std_msgs::msg::Float64>("rune/openvino/debug_pnp_x", rclcpp::SensorDataQoS());
    ros_pub_channels_->debug_pnp_y_pub_ =
      this->create_publisher<std_msgs::msg::Float64>("rune/openvino/debug_pnp_y", rclcpp::SensorDataQoS());
    ros_pub_channels_->debug_pnp_z_pub_ =
      this->create_publisher<std_msgs::msg::Float64>("rune/openvino/debug_pnp_z", rclcpp::SensorDataQoS());
    ros_pub_channels_->debug_now_angle_pub_ =
      this->create_publisher<std_msgs::msg::Float64>("rune/openvino/debug_now_angle", rclcpp::SensorDataQoS());
    if (!recorder_->open(ament_index_cpp::get_package_share_directory("rune") + "record.avi",
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         30.0,
                         cv::Size(camera_info_->image_width_, camera_info_->image_height_)))
    {
      RCLCPP_ERROR(this->get_logger(), "Recorder failed to open !");
    }
    RCLCPP_INFO(this->get_logger(), "End of Initializing debug-pub channels");
    using std::chrono_literals::operator""ms;
    time_ = this->create_wall_timer(100ms, std::bind(&OpenVINO_node::updateParams, this));
  } else
  {
    RCLCPP_WARN(this->get_logger(), "Release Mode");
    time_ = this->create_wall_timer(100ms, std::bind(&OpenVINO_node::updateParams, this));
  }
}

  void OpenVINO_node::updateNodeStatus()
  {
    const int node_status = this->get_parameter("NodeStatus").as_int();
    rune_status_str_ = this->get_parameter("autoaim_mode").as_string();
    const NodeStatus next_node_status =
      (node_status == 1) ? NodeStatus::ON_ACTIVATE : NodeStatus::ON_DEACTIVATE;
    if (node_status_ != next_node_status)
    {
      if (next_node_status == NodeStatus::ON_ACTIVATE)
      {
        ros_sub_channels_->img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          ros_sub_topic_->raw_img_topic_,
          rclcpp::SensorDataQoS(),
          std::bind(&OpenVINO_node::ImgCallback, this, std::placeholders::_1));

        ros_sub_channels_->camera_info_sub_ =
          this->create_subscription<sensor_msgs::msg::CameraInfo>(
            ros_sub_topic_->camera_info_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&OpenVINO_node::CameraInfoCallback, this, std::placeholders::_1));
      }
      else
      {
        ros_sub_channels_->img_sub_->clear_on_new_intra_process_message_callback();
        ros_sub_channels_->img_sub_.reset();
        ros_sub_channels_->camera_info_sub_->clear_on_new_intra_process_message_callback();
        ros_sub_channels_->camera_info_sub_.reset();

        RCLCPP_WARN(this->get_logger(), "OpenVINO node status is deactivated.");
      }
      node_status_ = next_node_status;
    }
  }

  OpenVINO_node::~OpenVINO_node()
  {
    RCLCPP_WARN(this->get_logger(), "Rune-OpenVINO recognitionNode ended ...");
  }

  void OpenVINO_node::ImgCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg){
    /* opencv camera-frame to ros2 camera-frame */
    const cv::Mat OpenCV_2_REP103 = (cv::Mat_<double>(3, 3) << 
       0, 0, 1, 
      -1, 0, 0, 
       0,-1, 0
    );
    const rclcpp::Time start_img_callback = this->now();

    *img_raw_ = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::cvtColor(*img_raw_, *img_bin_, cv::COLOR_BGR2GRAY);

    if(debug_)
    {
      *img_rec_ = img_raw_-> clone();
    }

    std::array<std::vector<rune::PosePoints>, NUM_CLASS> targets_info;
    try
    {
      // RCLCPP_INFO(this->get_logger(), "OpenVINO_node::detect() start ...");
      targets_info = openvino_engine_target_->detect(*img_raw_);
      // RCLCPP_INFO(this->get_logger(),"targets_info size: %d", targets_info[0].size());
      // RCLCPP_INFO(this->get_logger(),"fanblade_info size: %d", targets_info[1].size());

      if(targets_info[1].size() == 0 || targets_info[0].size() == 0) {
        RCLCPP_ERROR(this->get_logger(), "No rune detected");
        return;
      }
    } catch (const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "OpenVINO_node::detect() failed: %s", e.what());
      return;
    }

    if(!updateRecInfo(targets_info)){
      return;
    }

    fanblade_->center_ = targets_info[NetClass::TARGET][0].keypoints[FANBLADE_CENTER_INDEX];
  // TODO: 判断点的精确度，选点解pnp
    for(int i = 0; i < 4 ;i++) {
      if(i == FANBLADE_CENTER_INDEX) {
        continue; //跳过中心点
      }
      fanblade_->pnp_detection_points_3D_[i] = world_points_list_fanblade[i];
      fanblade_->pnp_detection_points_2D_[i] = cv::Point2f(targets_info[NetClass::TARGET][0].keypoints[i + 1].x, 
                                                            targets_info[NetClass::TARGET][0].keypoints[i + 1].y);

      if(rune_status_str_ == "big" && !pre_ready_) {
        fanblade_0_ -> pnp_detection_points_3D_[i] = world_points_list_fanblade[i];
        fanblade_0_ -> pnp_detection_points_2D_[i] = cv::Point2f(targets_info[fanblade_0_class_][fanblade_0_id_].keypoints[i + 1].x,
                                                                      targets_info[fanblade_0_class_][fanblade_0_id_].keypoints[i + 1].y);
      }
    }

    try {
      solvepose_target_->solvePnP(fanblade_->pnp_detection_points_3D_,
                           fanblade_->pnp_detection_points_2D_,
                           fanblade_->r_vec_,
                           fanblade_->t_vec_, *img_rec_);

      if(rune_status_str_ == "big" && !pre_ready_){
        solvepose_fanblade_0_->solvePnP(fanblade_0_->pnp_detection_points_3D_,
                               fanblade_0_->pnp_detection_points_2D_,
                               fanblade_0_->r_vec_,
                               fanblade_0_->t_vec_, *img_rec_);
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Rune-SolvePnp Error: %s", e.what());
        // 输出2D点和对应的3D点坐标
        std::cout << "fanblade_:"<<std::endl;
        for(int j = 0; j < 4; j++) {
          std::cout << "2D: " << fanblade_->pnp_detection_points_2D_[j].x << " " << fanblade_->pnp_detection_points_2D_[j].y << " 3D: " << fanblade_->pnp_detection_points_3D_[j].x << " " << fanblade_->pnp_detection_points_3D_[j].y << " " << fanblade_->pnp_detection_points_3D_[j].z << std::endl;
        }
        std::cout << "fanblade_0_:"<<std::endl;
        for(int j = 0; j < 4; j++) {
          std::cout << "2D: " << fanblade_0_->pnp_detection_points_2D_[j].x << " " << fanblade_0_->pnp_detection_points_2D_[j].y << " 3D: " << fanblade_0_->pnp_detection_points_3D_[j].x << " " << fanblade_0_->pnp_detection_points_3D_[j].y << " " << fanblade_0_->pnp_detection_points_3D_[j].z << std::endl;
        }
      return;
    }

    rune_sys_interfaces::msg::Fanblade fanblade_msg;

    /* RuneRotation: 0->NONE 1->CW 2->CCW */
    //目前识别效果对于识别旋转方向来说，比较不稳定，保险起见十帧取一个差值
    //这坨shit迟早删
    // openvino_engine_target_ -> now_vec_ = Eigen::Vector2f(fanblade_->pnp_detection_points_2D_[5].x-fanblade_->pnp_detection_points_2D_[7].x, fanblade_->pnp_detection_points_2D_[5].y-fanblade_->pnp_detection_points_2D_[7].y);
    // if(openvino_engine_target_->rotation_count_ == 10) {
    //   openvino_engine_target_->rotation_count_ = 0;
    //   openvino_engine_target_->rune_rotation_ = openvino_engine_target_->judgeRotation(openvino_engine_target_->last_vec_, openvino_engine_target_->now_vec_);
    //   openvino_engine_target_->last_vec_ = openvino_engine_target_->now_vec_;
    // }
    // openvino_engine_target_->rotation_count_+=1;

    // if (openvino_engine_target_->rune_rotation_ == RuneRotationStatue::NONE) {
    //   RCLCPP_WARN(this->get_logger(),"rune rotation state is none!");
    //   return;
    // }
    // else if (openvino_engine_target_->last_rune_rotation_ != openvino_engine_target_->rune_rotation_) {
    //   openvino_engine_target_->error_rotation_count_++;
    //   fanblade_msg.rotation = (openvino_engine_target_->last_rune_rotation_ == RuneRotationStatue::CW) ? 1u : 2u;
    //   if (openvino_engine_target_->error_rotation_count_ > 25) {
    //     RCLCPP_ERROR(this->get_logger(), "shit!");
    //     fanblade_msg.rotation = (openvino_engine_target_->rune_rotation_ == RuneRotationStatue::CW) ? 1u : 2u;
    //     openvino_engine_target_->last_rune_rotation_ = openvino_engine_target_->rune_rotation_;
    //     openvino_engine_target_->error_rotation_count_ = 0;
    //   }
    // }
    // else {
    //   openvino_engine_target_->error_rotation_count_ = 0;
    //   fanblade_msg.rotation = (openvino_engine_target_->rune_rotation_ == RuneRotationStatue::CW )? 1u : 2u;
    // }

    const cv::Mat t_vec_REP_103   = OpenCV_2_REP103 * fanblade_->t_vec_;
    const cv::Mat r_vec_REP_103   = OpenCV_2_REP103 * fanblade_->r_vec_;
    fanblade_->center_3D_.x       = t_vec_REP_103.at<double>(0) / 1.0e3;
    fanblade_->center_3D_.y       = t_vec_REP_103.at<double>(1) / 1.0e3;
    fanblade_->center_3D_.z       = t_vec_REP_103.at<double>(2) / 1.0e3;

    cv::Mat r_mat;
    cv::Rodrigues(r_vec_REP_103, r_mat);

    fanblade_msg.header.stamp               = msg->header.stamp;
    fanblade_msg.header.frame_id            = "camera_frame";
    fanblade_msg.fanblade_center.x = fanblade_->center_3D_.x;
    fanblade_msg.fanblade_center.y = fanblade_->center_3D_.y;
    fanblade_msg.fanblade_center.z = fanblade_->center_3D_.z;
    Eigen::Matrix3d rotation_matrix{
      {r_mat.at<double>(0, 0), r_mat.at<double>(0, 1), r_mat.at<double>(0, 2)},
      {r_mat.at<double>(1, 0), r_mat.at<double>(1, 1), r_mat.at<double>(1, 2)},
      {r_mat.at<double>(2, 0), r_mat.at<double>(2, 1), r_mat.at<double>(2, 2)}
    };
    Eigen::Matrix<double, 3, 1> translation_vec(t_vec_REP_103.at<double>(0, 0),
                                                t_vec_REP_103.at<double>(1, 0),
                                                t_vec_REP_103.at<double>(2, 0));

    fanblade_msg.t_vec[0]            = t_vec_REP_103.at<double>(0, 0);
    fanblade_msg.t_vec[1]            = t_vec_REP_103.at<double>(1, 0);
    fanblade_msg.t_vec[2]            = t_vec_REP_103.at<double>(2, 0);
    tf2::Matrix3x3 tf2_rot_mat(
      r_mat.at<double>(0, 0), r_mat.at<double>(0, 1), r_mat.at<double>(0, 2),
      r_mat.at<double>(1, 0), r_mat.at<double>(1, 1), r_mat.at<double>(1, 2),
      r_mat.at<double>(2, 0), r_mat.at<double>(2, 1), r_mat.at<double>(2, 2)
    );
    tf2::Quaternion tf2_quaternion;
    tf2_rot_mat.getRotation(tf2_quaternion);
    fanblade_msg.rotation  = (rune_rotation_statue_ == RuneRotationStatue::CW )? 1u : 2u;
    double r,p,y;
    tf2_rot_mat.getRPY(r,p,y); 
    r+=M_PI;
    // openvino_engine_target_->updateAngle(r);
    fanblade_msg.now_angle = r;
    
    if(rune_status_str_ == "big" && !pre_ready_) {
      const cv::Mat r_vec_REP_103_0 = OpenCV_2_REP103 * fanblade_0_->r_vec_;
      cv::Mat r_mat_0;
      cv::Rodrigues(r_vec_REP_103_0, r_mat_0);  
      tf2::Matrix3x3 tf2_rot_mat_0(
        r_mat_0.at<double>(0, 0), r_mat_0.at<double>(0, 1), r_mat_0.at<double>(0, 2),
        r_mat_0.at<double>(1, 0), r_mat_0.at<double>(1, 1), r_mat_0.at<double>(1, 2),
        r_mat_0.at<double>(2, 0), r_mat_0.at<double>(2, 1), r_mat_0.at<double>(2, 2)
      );
      double roll, pitch, yaw;
      tf2_rot_mat_0.getRPY(roll, pitch, yaw);
      fanblade_msg.fanblade_angle_0 = roll + M_PI;
    }

    if(debug_) {
      std_msgs::msg::Float64 debug_dif_angle_msg;
      debug_dif_angle_msg.data = openvino_engine_target_ -> dif_angle_;
      ros_pub_channels_->debug_dif_angle_->publish(debug_dif_angle_msg);
    }

    fanblade_msg.fanblade_now_id = openvino_engine_target_ -> fanblade_id_;
    // RCLCPP_INFO(this->get_logger(), "now_id: %d, angle_0: %f", fanblade_msg.fanblade_now_id, fanblade_msg.fanblade_angle_0);

    fanblade_msg.quaternion.orientation = tf2::toMsg(tf2_quaternion);

    const auto end_img_callback = this->now();
    timecost_->ImgCallback_cost =
      (end_img_callback - start_img_callback).to_chrono<std::chrono::milliseconds>();

    // /* send rune-fanblade msg */
    ros_pub_channels_->fanblade_pub_->publish(fanblade_msg);
    // RCLCPP_INFO(this->get_logger(), "send fanblade msg");

    if(debug_) {
      if (debug_params_->show_pnp) {
        std_msgs::msg::Float64 pnp_x_msg;
        pnp_x_msg.data = fanblade_->center_3D_.x;
        std_msgs::msg::Float64 pnp_y_msg;
        pnp_y_msg.data = fanblade_->center_3D_.y;
        std_msgs::msg::Float64 pnp_z_msg;
        pnp_z_msg.data = fanblade_->center_3D_.z;
        ros_pub_channels_->debug_pnp_x_pub_->publish(pnp_x_msg);
        ros_pub_channels_->debug_pnp_y_pub_->publish(pnp_y_msg);
        ros_pub_channels_->debug_pnp_z_pub_->publish(pnp_z_msg);
        cv::drawFrameAxes(*img_rec_,
                          camera_info_->camera_matrix_,
                          camera_info_->dist_coeffs_,
                          fanblade_->r_vec_,
                          fanblade_->t_vec_,
                          50,
                          7);
        if(rune_status_str_ == "big"&& !pre_ready_){
          cv::drawFrameAxes(*img_rec_,
                          camera_info_->camera_matrix_,
                          camera_info_->dist_coeffs_,
                          fanblade_0_->r_vec_,
                          fanblade_0_->t_vec_,
                          50,
                          7);
        }
      }
      if(debug_params_->rec_img_){
        openvino_engine_target_->draw_pose(*img_rec_, targets_info);
        // RCLCPP_INFO(this->get_logger(), "draw_pose success");
        //在img_rec_图像上显示世界坐标系与2D点的对应
        // for (int i = 0; i < 4; i++) {
        //   //绘制每个点的2D坐标
        //   cv::circle(*img_rec_, fanblade_->pnp_detection_points_2D_[i], 5, cv::Scalar(0, 0, 255), 2);
        //   cv::putText(*img_rec_, std::to_string(fanblade_->pnp_detection_points_3D_[i].x) + " " + std::to_string(fanblade_->pnp_detection_points_3D_[i].y) + " " + std::to_string(fanblade_->pnp_detection_points_3D_[i].z), fanblade_->pnp_detection_points_2D_[i], cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::rectangle(*img_rec_, r_detecter_->R_->Rect_, cv::Scalar(0, 255, 0), 2);
        ros_pub_channels_->debug_rec_img_pub_.publish(
          cv_bridge::CvImage(msg->header, "bgr8", *img_rec_).toImageMsg());
      }
    }
    // RCLCPP_INFO(this->get_logger(), "end");
  }

  void OpenVINO_node::CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
  {
    camera_info_->camera_matrix_.at<double>(0, 0) = camera_info_msg->k[0];
    camera_info_->camera_matrix_.at<double>(0, 1) = camera_info_msg->k[1];
    camera_info_->camera_matrix_.at<double>(0, 2) = camera_info_msg->k[2];
    camera_info_->camera_matrix_.at<double>(1, 0) = camera_info_msg->k[3];
    camera_info_->camera_matrix_.at<double>(1, 1) = camera_info_msg->k[4];
    camera_info_->camera_matrix_.at<double>(1, 2) = camera_info_msg->k[5];
    camera_info_->camera_matrix_.at<double>(2, 0) = camera_info_msg->k[6];
    camera_info_->camera_matrix_.at<double>(2, 1) = camera_info_msg->k[7];
    camera_info_->camera_matrix_.at<double>(2, 2) = camera_info_msg->k[8];
  
    camera_info_->dist_coeffs_  = camera_info_msg->d;
    camera_info_->image_width_  = camera_info_msg->width;
    camera_info_->image_height_ = camera_info_msg->height;

    solvepose_target_->update(camera_info_->camera_matrix_, camera_info_->dist_coeffs_);
    solvepose_fanblade_0_->update(camera_info_->camera_matrix_, camera_info_->dist_coeffs_);
  }

  void OpenVINO_node::preReadyCallback(const std_msgs::msg::Bool::ConstSharedPtr msg) {
    pre_ready_ = msg->data;
    // RCLCPP_INFO(this->get_logger(), "pre_ready: %d", pre_ready_);
  }

  bool OpenVINO_node::updateRecInfo(std::array<std::vector<PosePoints>, NUM_CLASS>& points_info) {
    Eigen::Vector2f now_vec;
    cv::Point2f p1_now = (points_info[NetClass::TARGET][0].keypoints[1] + points_info[NetClass::TARGET][0].keypoints[4]) / 2.0;
    cv::Point2f p2_now = (points_info[NetClass::TARGET][0].keypoints[2] + points_info[NetClass::TARGET][0].keypoints[3]) / 2.0;
    now_vec = Eigen::Vector2f(p2_now.x - p1_now.x, p2_now.y - p1_now.y);
    if(is_first_frame_) {
        last_fanblade_0_vecs_.push_back(now_vec);
        last_fanblade_0_vec_ = now_vec;
        last_5_fanblade_vec_  = now_vec;
        last_5_fanblade_0_vec_ = now_vec;
        check_rotation_count_++;
        is_first_frame_ = false;
        return false;
    }
    else if(rune_status_str_ == "big" && !pre_ready_) {
      //通过与上一帧的fanblade_0方向的夹角判断fanblade_0的位置
      int min_class = 0;
      int min_id = 0;
      float min_angle = 6*M_PI;
      for(int i = 0; i < NUM_CLASS; i++) {
        int num_objects = points_info[i].size();
        for(int j = 0; j < num_objects; j++) {
          cv::Point2f p1 = (points_info[i][j].keypoints[1] + points_info[i][j].keypoints[4]) / 2.0;
          cv::Point2f p2 = (points_info[i][j].keypoints[2] + points_info[i][j].keypoints[3]) / 2.0;
          if(debug_ && debug_params_-> rec_img_){
            cv::line(*img_rec_, p1, p2, cv::Scalar(0, 0, 255), 2);
          }
          Eigen::Vector2f now_vec(p2.x - p1.x, p2.y - p1.y);
          float angle;
          // for(auto last_vec : last_fanblade_0_vecs_) {
          //   float cos_theta = last_vec.dot(now_vec) / (last_vec.norm() * now_vec.norm());
          //   angle += acos(cos_theta);
          // }
          angle = acos(now_vec.dot(last_fanblade_0_vec_) / (now_vec.norm() * last_fanblade_0_vec_.norm()));

          if(angle < min_angle) {
            min_angle = angle;
            min_class = i;
            min_id = j;
          }
        }
      }

      if(min_angle > 40 / 180.0 * M_PI) {
        RCLCPP_INFO(this->get_logger(), "lost fanblade_0");
        return false;
      }
      fanblade_0_class_ = min_class;
      fanblade_0_id_ = min_id;
      cv::Point2f p_1 = (points_info[min_class][min_id].keypoints[1] + points_info[min_class][min_id].keypoints[4]) / 2.0;
      cv::Point2f p_2 = (points_info[min_class][min_id].keypoints[2] + points_info[min_class][min_id].keypoints[3]) / 2.0;
      if(debug_ && debug_params_-> rec_img_) {
        cv::line(*img_rec_, p_1, p_2, cv::Scalar(0, 255, 0), 2);
         // for(auto last_vec : last_fanblade_0_vecs_){
         //   cv::line(*img_rec_, p_1, p_1 + cv::Point2f(last_vec.x(), last_vec.y()), cv::Scalar(255, 0, 0), 2);
         // }
      }
      last_fanblade_0_vec_ = Eigen::Vector2f(p_2.x - p_1.x, p_2.y - p_1.y);
    }
    //利用靶标两帧夹角判断旋转方向以及叶片是否切换
    double cross = now_vec.x()*last_5_fanblade_vec_.y() - now_vec.y() * last_5_fanblade_vec_.x();
    double angle = asin(cross / (now_vec.norm() * last_5_fanblade_vec_.norm()));
    if(angle > 20 / 180.0 * M_PI) {
      openvino_engine_target_ -> fanblade_id_ = (openvino_engine_target_ -> fanblade_id_ + 1) % 4;
    }
    RuneRotationStatue rune_rotation_statue;
    if(cross < 0) {
      rune_rotation_statue = RuneRotationStatue::CW; 
    }
    else rune_rotation_statue = RuneRotationStatue::CCW;

    if(last_rune_rotation_statue_ == RuneRotationStatue::NONE) {
      error_rotation_count_ = 0;
      rune_rotation_statue_ = rune_rotation_statue;
      last_rune_rotation_statue_ = rune_rotation_statue;
    }
    else if (last_rune_rotation_statue_ == rune_rotation_statue) {
      error_rotation_count_ = 0;
      rune_rotation_statue_ = rune_rotation_statue;
    }
    else if (last_rune_rotation_statue_ != rune_rotation_statue) {
      error_rotation_count_ ++;
      if(error_rotation_count_ > 15) {
        error_rotation_count_ = 13;
        RCLCPP_INFO(this->get_logger(), "error_ratation : %d", error_rotation_count_);
        last_rune_rotation_statue_ = rune_rotation_statue;
        rune_rotation_statue_      = rune_rotation_statue;
      }
      else rune_rotation_statue_ = last_rune_rotation_statue_;
    }
    if(rotation_frame_count_ == 5) {
      rotation_frame_count_ = 0;
      last_5_fanblade_vec_ = now_vec;
    }
    rotation_frame_count_++;
    
    return true;
  }

  void OpenVINO_node::judgeRotation(double angle) {
    if(last_fanblade_0_angle_ == 0) {
      last_fanblade_0_angle_ = angle;
      return;
    }
    double angle_diff = angle - last_fanblade_0_angle_;
    RuneRotationStatue rune_rotation = RuneRotationStatue::NONE;
    if(angle_diff > 0) {
      rune_rotation = RuneRotationStatue::CCW;
    }
    else if(angle_diff < 0) {
      rune_rotation = RuneRotationStatue::CW;
    }

    if(rune_rotation != last_rune_rotation_statue_) {
      error_rotation_count_++;
      if(error_rotation_count_ > 0) {
        rune_rotation_statue_ = rune_rotation;
        last_rune_rotation_statue_ = rune_rotation;
      }
      else {
        rune_rotation_statue_=  last_rune_rotation_statue_;
      }
    }
    else if(rune_rotation == last_rune_rotation_statue_) {
      error_rotation_count_ --;
      rune_rotation_statue_ = rune_rotation;
    }
    last_fanblade_0_angle_ = angle;
  }
}; // namespace rune
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rune::OpenVINO_node)