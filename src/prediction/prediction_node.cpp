#include "prediction/prediction_node.hpp"

#include <rclcpp/node_options.hpp>
#include <rune_sys_interfaces/msg/detail/fanblade__struct.hpp>

namespace rune
{
PredictionNode::PredictionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PredictionNode", options)
{
  declareParams();
  RCLCPP_WARN(this->get_logger(), "Rune-PredictionNode start ...");
  RCLCPP_WARN(this->get_logger(), "Fanblades-Buffer queue length: %d", QUEUE_LENGTH);

  std::unique_ptr<RuneOdomSolver> rune_odom_solver = std::make_unique<RuneOdomSolver>(QUEUE_LENGTH);
  rune_bullet_solve_ =  std::make_shared<rm::NewBulletSolve<double, true, false>>(rune_bullet_params_);

  sub_fanblade_ = this->create_subscription<rune_sys_interfaces::msg::Fanblade>(
    fanblade_sub_topic_name_, 10,
    std::bind(&PredictionNode::fanbladeCallback, this, std::placeholders::_1));

  pub_target_ =
    this->create_publisher<rune_sys_interfaces::msg::Target>(target_pub_topic_name_, 10);
  
  pub_pre_ready_ =
    this->create_publisher<std_msgs::msg::Bool>(pre_ready_pub_name_, 10);

  pub_allow_shoot_ = 
    this->create_publisher<std_msgs::msg::Float64>(allow_shoot_pub_name_, rclcpp::SensorDataQoS());

  using std::chrono_literals::operator""ms;
  status_timer_ = 
    this->create_wall_timer(200ms, std::bind(&PredictionNode::getNodeStatus, this));
  const int node_status = this->declare_parameter<int>("NodeStatus", 1);

  rune_statue_str_ = this->declare_parameter<std::string>("autoaim_mode", "normal");

  if (debug_)
  {
    pub_debug_angle_ = this->create_publisher<std_msgs::msg::Float64>(
      angle_debug_pub_topic_name_, rclcpp::SensorDataQoS());
    pub_debug_omega_ = this->create_publisher<std_msgs::msg::Float64>(
      omega_debug_pub_topic_name_, rclcpp::SensorDataQoS());
    pub_debug_pre_x_ = this->create_publisher<std_msgs::msg::Float64>(
      pre_x_debug_pub_name_, rclcpp::SensorDataQoS());
    pub_debug_pre_y_ = this->create_publisher<std_msgs::msg::Float64>(
      pre_y_debug_pub_name_, rclcpp::SensorDataQoS());
    pub_debug_pre_z_ = this->create_publisher<std_msgs::msg::Float64>(
      pre_z_debug_pub_name_, rclcpp::SensorDataQoS());

    odom_fanblade_x_pub_ = 
    this->create_publisher<std_msgs::msg::Float64>("/rune/debug/odom/fanblade/x", 10);

    odom_fanblade_y_pub_ = 
      this->create_publisher<std_msgs::msg::Float64>("/rune/debug/fanblade/odom/y", 10);

    odom_fanblade_z_pub_ = 
      this->create_publisher<std_msgs::msg::Float64>("/rune/debug/odom/fanblade/z", 10);

    pre_angle_pub_ = 
      this->create_publisher<std_msgs::msg::Float64>("/rune/pre_angle", 10);

    now_angle_pub_ =
      this->create_publisher<std_msgs::msg::Float64>("/rune/now_angle", 10);
  }
}

PredictionNode::~PredictionNode()
{
  RCLCPP_WARN(this->get_logger(), "Rune-PredictionNode ended ...");
}

void PredictionNode::getNodeStatus() {
  rune_statue_str_ = this->get_parameter("autoaim_mode").as_string();
  updateParams();
  //  RCLCPP_INFO(this->get_logger(), "shit! rune_statue_str_: %s, last_rune_state_str_: %s", rune_statue_str_.c_str(), last_rune_state_str_.c_str());
  try
  {
    if(last_rune_state_str_ != rune_statue_str_ && (last_rune_state_str_ == "big" || last_rune_state_str_ == "small")) {
      // 重置预测器
      RCLCPP_ERROR(this->get_logger(), "rune exit, reset predictor_ceres_");
      reset();
      predictor_ceres_ -> reset();  
    }
    last_rune_state_str_ = rune_statue_str_;

    std_msgs::msg::Bool pre_ready;
    pre_ready.data = pre_ready_;
    pub_pre_ready_->publish(pre_ready);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(this->get_logger(), "Get autoaim_mode error: %s", e.what());
  }
  // RCLCPP_INFO(this->get_logger(), "rune-statue: %s, last_rune_state_str_: %s", rune_statue_str_.c_str(), last_rune_state_str_.c_str());
}

void PredictionNode::fanbladeCallback(
  const rune_sys_interfaces::msg::Fanblade::ConstSharedPtr & fanblade_msg)
{
  // RCLCPP_INFO(this->get_logger(), "fanbladeCallback start ...");
  if (error_prediction_count_ >= 10)
  {
    reset();
    error_prediction_count_ = 0;
    RCLCPP_WARN(this->get_logger(), "error prediction count >= 10, it will reset.");
    return;
  }

  /* -------------------------------------- Judge Rune Statue ------------------------------------- */
  // RCLCPP_INFO(this->get_logger(), "rune-statue: %s", rune_statue_str_.c_str());
  rune_statue_str_ = "small";
  if (rune_statue_str_ == "normal" || rune_statue_str_ == "autoaim")
  {
    return;
  }
  else if (rune_statue_str_ == "big" || rune_statue_str_ == "small")
  {
    rune_statue_ = rune_statue_str_ == "big" ? RuneStatue::BIG : RuneStatue::SMALL;
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Autoaim Error!");
  }

  /* ---------------------------------- Deal with Angle and time ---------------------------------- */
  rune_rotation_statue_ = fanblade_msg->rotation == 1u ? RuneRotationStatue::CW : RuneRotationStatue::CCW;
  now_angle_            = fanblade_msg->now_angle;
  fanblade_id_          = fanblade_msg->fanblade_now_id;

  // RCLCPP_INFO(this->get_logger(), "now angle: %f", fanblade_msg->now_angle*180/M_PI);

  const Eigen::Vector3d fanblade_center_odom = {fanblade_msg->fanblade_center.x, 
                                                fanblade_msg->fanblade_center.y, 
                                                fanblade_msg->fanblade_center.z};

  //5.3日debug用
  if(debug_) {
    std_msgs::msg::Float64 fanblade_x;
    fanblade_x.data = fanblade_center_odom.x();
    odom_fanblade_x_pub_ -> publish(fanblade_x);

    std_msgs::msg::Float64 fanblade_y;
    fanblade_y.data = fanblade_center_odom.y();
    odom_fanblade_y_pub_ -> publish(fanblade_y);

    std_msgs::msg::Float64 fanblade_z;
    fanblade_z.data = fanblade_center_odom.z();
    odom_fanblade_z_pub_ -> publish(fanblade_z);
  }

  /* ROS2 time stamp to s */
  now_time_ = fanblade_msg->header.stamp.sec + fanblade_msg->header.stamp.nanosec * 1.0e-9;
  if(rune_rotation_statue_ == RuneRotationStatue::CW) {
    angle_ = 2*M_PI - fanblade_msg->fanblade_angle_0;
  }
  else {
    angle_ = fanblade_msg->fanblade_angle_0;
  }

  if(!pre_init_){
    begin_time_ = now_time_;
    pre_init_ = true;
    last_angle_ = angle_;
    deque_count_threshold_ = BUFFER_LENGTH/QUEUE_LENGTH;
  }

  if(rune_statue_str_ == "big") {
    checkAngle();
    // RCLCPP_INFO(this->get_logger(), "big-rune prediction start, now_time: %lf, angle: %lf", now_time_ -begin_time_, angle_);
    fanblade_id_ = fanblade_msg-> fanblade_now_id;

    double angle_tmp = angle_ + 2* M_PI* jump_transformation_count_;
    // RCLCPP_INFO(this->get_logger(), "jump_transformation_count_: %d, angle_tmp: %lf", jump_transformation_count_, angle_tmp);
    // 存入时间和角度
    if(time_sequence_ -> size() != QUEUE_LENGTH && last_angle_ != angle_ && buffer_count_ < deque_count_threshold_) {
      time_sequence_ -> push_back(now_time_ - begin_time_);

      angle_sequence_ -> push_back(angle_tmp);

      last_angle_ = angle_;
      if(buffer_count_ == 0) return;
    }
    else if(time_sequence_ -> size() == QUEUE_LENGTH & buffer_count_ < deque_count_threshold_) {
      // RCLCPP_INFO(this->get_logger(), "buffer_count_ == %d, now_time: %lf, begin_time: %lf", buffer_count_, now_time_ - begin_time_, begin_time_);
      std::copy(time_sequence_ -> begin(), time_sequence_ -> end(), time_buffer_ -> begin() + buffer_count_ * QUEUE_LENGTH);
      std::copy(angle_sequence_ -> begin(), angle_sequence_ -> end(), angle_buffer_ -> begin() + buffer_count_ * QUEUE_LENGTH);

      // RCLCPP_INFO(this->get_logger(), "buffer_count_ == %d, buffering size %d...", buffer_count_, angle_buffer_ -> size());
      buffer_count_++;
      time_sequence_ ->clear();
      angle_sequence_ ->clear();

      time_sequence_ -> push_back(now_time_ - begin_time_);
      angle_sequence_ -> push_back(angle_tmp);

      last_angle_ = angle_;
      return;
    }
    else if(buffer_count_ == deque_count_threshold_) {
      // RCLCPP_INFO(this->get_logger(), "buffer_count_ == 3, buffering size %d...", angle_buffer_ -> size());
      buffer_count_ += 1;

      // for (int i = 0; i < BUFFER_LENGTH; i++){
      //   //打印buffer
      //   RCLCPP_INFO(this->get_logger(),"buffer time: %lf, angle: %lf", (*time_buffer_)[i], (*angle_buffer_)[i]);
      // }
      // RCLCPP_INFO(this->get_logger(), "buffer_count_ == 3, ori data size: %d、 now data size: %d,  predicting start...", ori_data_size, angle_buffer_ -> size());
    }
    if (buffer_count_ < deque_count_threshold_) return;

    /* ---------------------------------- Verify data availability ---------------------------------- */
    int zero_count = std::count(time_buffer_->begin(), time_buffer_->end(), 0.0);
    if (zero_count > 1)
    {
      RCLCPP_WARN(this->get_logger(), "Buffer has zero data, zore count: %d, pass predict.", zero_count);
      return;
    }

    if (angle_buffer_->size() != BUFFER_LENGTH && rune_statue_ == RuneStatue::BIG)
    {
      RCLCPP_INFO(this->get_logger(), "Buffer-length isn't full, pass predict.");
      return;
    }
  }

  /* -------------------------------------- Start predicting -------------------------------------- */
  std::copy(fanblade_msg->t_vec.begin(), fanblade_msg->t_vec.end(), world2camera_t_vec_.begin());

  // for (int i = 0; i < QUEUE_LENGTH; i++)
  // {
  //   RCLCPP_INFO(
  //     this->get_logger(), "time: %lf, angle: %lf", (*time_buffer_)[i], (*angle_buffer_)[i]);
  // }
  // RCLCPP_INFO(this->get_logger(), "last time: %lf", (*time_buffer_)[QUEUE_LENGTH - 1]);
  predictor_ceres_->update((*angle_buffer_), (*time_buffer_));

  // RCLCPP_INFO(this->get_logger(), "enter predict, bullet_params: %lf, %lf, %lf, %lf, %lf", rune_bullet_params_->bullet_m, rune_bullet_params_->bullet_r, rune_bullet_params_->bullet_v, rune_bullet_params_-> g, rune_bullet_params_-> mu);
  const auto horizontal_dist = std::sqrt(fanblade_msg->fanblade_center.x * fanblade_msg->fanblade_center.x +
                                              fanblade_msg->fanblade_center.y * fanblade_msg->fanblade_center.y);
  // const auto target_Rune_p = rm::NewBulletSolve<double, true, true>::Point2D{horizontal_dist, fanblade_msg->fanblade_center.z};
  const auto target_Rune_p = rm::NewBulletSolve<double, true, true>::Point2D{4.8, 3.8};
  // RCLCPP_INFO(this->get_logger(), "1");
  const auto calcresult_Rune = rune_bullet_solve_->calcTheta(target_Rune_p);
  bullet_fly_time_ = calcresult_Rune->bullet_flight_time.count();
  bullet_fly_time_ = bullet_fly_time_ + predict_params_ -> time_offset_; /**< s*/
  double pitch = calcresult_Rune->theta;

  // RCLCPP_INFO(this->get_logger(), "2");
  RCLCPP_INFO(this->get_logger(), "bullet_fly_time: %lf s, offset: %lf, pitch: %lf", bullet_fly_time_, predict_params_ -> time_offset_, pitch);
  // RCLCPP_INFO(this->get_logger(), "prediction init time: %lf", (*time_buffer_)[QUEUE_LENGTH - 1]);

  try
  {
    // pre_angle_ = predictor_ceres_->predict((*time_buffer_)[QUEUE_LENGTH - 1],
    //                                       bullet_fly_time_ ,
    //                                       rune_statue_,
    //                           (*angle_buffer_)[QUEUE_LENGTH - 1]);
    if(rune_statue_str_ == "big") {
      pre_angle_ = predictor_ceres_ -> bigPredict (now_time_ -begin_time_, bullet_fly_time_, angle_);

      if(!predictor_ceres_->big_rune_pre_ready_) {
        RCLCPP_INFO(this->get_logger(), "reset predictor_ceres_");
        reset();
        return;
      }
      // RCLCPP_INFO(this->get_logger(), "big-rune prediction end, pre_angle_: %lf, now_angle_: %lf, now_time_ :%lf, bullet_fly_time_: %lf", pre_angle_ / M_PI * 180.0, angle_ / M_PI * 180.0, now_time_ - begin_time_, bullet_fly_time_);
    }
    else if(rune_statue_str_ == "small") {
      pre_angle_ = predictor_ceres_ -> smallPredict(bullet_fly_time_);
      RCLCPP_INFO(this->get_logger(), "pre_angle: %lf", pre_angle_);
    }
    // RCLCPP_INFO(this->get_logger(), "i{it angle: %lf", (*angle_buffer_)[QUEUE_LENGTH - 1]);
    if (!std::isnormal(pre_angle_))
    {
      RCLCPP_WARN(this->get_logger(), "pre_angle_ is not normal.");
      error_prediction_count_++;
      return;
    }
    if (pre_angle_ > 180.0 / 180.0 * M_PI)
    {
      predictor_ceres_-> big_rune_pre_ready_ = false;
      RCLCPP_WARN(this->get_logger(), "pre_angle_ is too big, pre_angle: %lf, now_angle: %lf, begin_time: %lf, now_time: %lf, pre_time: %lf,  it will reset!", pre_angle_ / M_PI * 180.0, angle_ / M_PI * 180.0, begin_time_, now_time_-begin_time_, now_time_ - begin_time_ + bullet_fly_time_);
      error_prediction_count_++;
      return;
    }
    checkShoot();

    if(debug_) {
      std_msgs::msg::Float64 pre_angle ;
      float pre_angle_location = pre_angle_ / M_PI * 180.0 + now_angle_ / M_PI * 180.0;
      if(pre_angle_location>360){
        pre_angle_location -= 360;
      }
      pre_angle.data = pre_angle_location;
      pre_angle_pub_->publish(pre_angle); 

      std_msgs::msg::Float64 now_angle_msg;
      float now_angle_location =  now_angle_ / M_PI * 180.0;
      now_angle_msg.data = now_angle_location;
      now_angle_pub_->publish(now_angle_msg);
    }

    // RCLCPP_INFO(this->get_logger(), "pre-angle: %lf", pre_angle_ / M_PI * 180.0);
    // RCLCPP_INFO(this->get_logger(), "cost-time: %ld", predictor_ceres->getTimeCost());
    // RCLCPP_INFO(this->get_logger(), "omega: %lf", predictor_ceres->getOmega());
    // double params[5];
    // predictor_ceres->getParams(params);
    // RCLCPP_INFO(this->get_logger(),
    //             "params: %lf, %lf, %lf, %lf, %lf",
    //             params[0],
    //             params[1],
    //             params[2],
    //             params[3],
    //             params[4]);
    
    inferencePose(target_pose_worldframe_, pre_angle_);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(this->get_logger(), "Rune-PredictionNode Error: %s", e.what());
    return;
  }

  pre_ready_ = true;

  /* --------------------------------- world_frame to camera_frame -------------------------------- */
  const Eigen::Quaternion world2camera (fanblade_msg->quaternion.orientation.w,
                                        fanblade_msg->quaternion.orientation.x,
                                        fanblade_msg->quaternion.orientation.y,
                                        fanblade_msg->quaternion.orientation.z);
  const Eigen::Matrix3d world2camera_rotation = world2camera.toRotationMatrix();
  Eigen::Matrix<double, 3, 1> world_point {target_pose_worldframe_.x,
                                           target_pose_worldframe_.y,
                                           target_pose_worldframe_.z};
  const Eigen::Matrix<double, 3, 3> OpenCV_2_REP103 {{ 0, 0, 1},
                                                     {-1, 0, 0},
                                                     { 0,-1, 0}};
  world_point = OpenCV_2_REP103 * world_point;
  const Eigen::Matrix<double, 3, 1> t {world2camera_t_vec_[0],
                                       world2camera_t_vec_[1],
                                       world2camera_t_vec_[2]};
  Eigen::Matrix<double, 3, 1> target_pose_cameraframe = world2camera_rotation * world_point + t;

  target_pose_cameraframe[0] /= 1.0e3;
  target_pose_cameraframe[1] /= 1.0e3;
  target_pose_cameraframe[2] /= 1.0e3;

  /* ------------------------------------- building Target-msg ------------------------------------ */
  rune_sys_interfaces::msg::Target target_msg;
  target_msg.header.stamp = fanblade_msg->header.stamp;
  target_msg.header.frame_id = "camera_frame";
  target_msg.angle = pre_angle_ + now_angle_;
  target_msg.is_tracked = true;
  target_msg.pose.position.x = target_pose_cameraframe(0);
  target_msg.pose.position.y = target_pose_cameraframe(1);
  target_msg.pose.position.z = target_pose_cameraframe(2);
  std_msgs::msg::Float64 allow_shoot;
  allow_shoot.data = (allow_shoot_ == true)? 1.0 : 0.0;
  if(allow_shoot_) {
    allow_shoot_ = false;
  }
  pub_target_->publish(target_msg);
  pub_allow_shoot_->publish(allow_shoot);

  /* -------------------------------------------- debug ------------------------------------------- */
  if (debug_)
  {
    std_msgs::msg::Float64 omega_msg;
    omega_msg.data = predictor_ceres_->getOmega();
    pub_debug_omega_->publish(omega_msg);
    std_msgs::msg::Float64 pre_x;
    pre_x.data = target_pose_cameraframe[0];
    pub_debug_pre_x_->publish(pre_x);
    std_msgs::msg::Float64 pre_y;
    pre_y.data = target_pose_cameraframe[1];
    pub_debug_pre_y_->publish(pre_y);
    std_msgs::msg::Float64 pre_z;
    pre_z.data = target_pose_cameraframe[2];
    pub_debug_pre_z_->publish(pre_z);
    std_msgs::msg::Float64 angle_msg;
    angle_msg.data = angle_;
    pub_debug_angle_->publish(angle_msg);
    if (debug_record_data_)
    {
      recordData("./big-rune.csv", *time_buffer_, *angle_buffer_);
    }
  }
}

void PredictionNode::inferencePose(cv::Point3f & target_pose, const double dif_angle)
{
  const double angle = std::abs(dif_angle);
  double x     = predict_params_->rune_radius_ * std::sin(angle);
  double y     = predict_params_->rune_radius_ * (1 - std::cos(angle));

  // RCLCPP_INFO(this->get_logger(), "rotation: %d", rune_rotation_statue_);
  if (rune_rotation_statue_ == RuneRotationStatue::CW)
  {
    x = -x;
  }

  target_pose.x = x;
  target_pose.y = y;
  target_pose.z = 0;
}

}  // namespace rune

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rune::PredictionNode)