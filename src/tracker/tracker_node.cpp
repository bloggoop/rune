#include "tracker/tracker_node.hpp"

namespace rune
{

TrackerNode::TrackerNode(const rclcpp::NodeOptions & options) : Node("TrackerNode", options)
{
  RCLCPP_INFO(get_logger(), "Rune TrackerNode Start...");

  declareParams();

  // subscribe
  camera_target_sub_.subscribe(this, sub_camera_target_topic_);
  camera2odom_sub_.subscribe(this, sub_camera2odom_topic_);
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), camera_target_sub_, camera2odom_sub_);
  sync_->registerCallback(&TrackerNode::TargetCallback, this);

  pub_tracked_target_ =
    this->create_publisher<autoaim_sys_interfaces::msg::TrackedRune>(pub_odom_tracker_target_, rclcpp::SensorDataQoS());
  pub_debug_x = 
    this->create_publisher<std_msgs::msg::Float64> (pub_debug_x_topic, 10);
  pub_debug_y = 
    this->create_publisher<std_msgs::msg::Float64> (pub_debug_y_topic, 10);
  pub_debug_z = 
    this->create_publisher<std_msgs::msg::Float64> (pub_debug_z_topic, 10);
  pub_debug_before_x = 
    this->create_publisher<std_msgs::msg::Float64> (pub_odom_tracker_x, 10);
  pub_debug_before_y =
    this->create_publisher<std_msgs::msg::Float64> (pub_odom_tracker_y, 10);
  pub_debug_before_z = 
    this->create_publisher<std_msgs::msg::Float64> (pub_odom_tracker_z, 10);
  
  pub_debug_chi_ = this->create_publisher<std_msgs::msg::Float64>("debug_rune_chi", 10);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  using std::chrono_literals::operator""ms;
  update_params_timer_ =
    this->create_wall_timer(500ms, std::bind(&TrackerNode::updateParams, this));
  
  RCLCPP_INFO(this->get_logger(), "Rune TrackerNode over init...");

  // debug
  pub_odom_target_debug_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/rune/tracker/debug/odom_target", rclcpp::SensorDataQoS());
}

TrackerNode::~TrackerNode() { RCLCPP_INFO(this->get_logger(), "Rune TrackerNode Shutdown..."); }

void TrackerNode::TargetCallback(
  const TargetMsg::ConstSharedPtr & target_msg,
  const TransformStamped::ConstSharedPtr & camera2odom_transform)
{
  geometry_msgs::msg::Pose odom_point;
  geometry_msgs::msg::Pose camera_point;
  camera_point = target_msg->pose;
  tf2::doTransform(camera_point, odom_point, *camera2odom_transform);

  std_msgs::msg::Float64 x_before, y_before, z_before;
  x_before.data = odom_point.position.x;
  pub_debug_before_x -> publish(x_before);

  y_before.data = odom_point.position.y;
  pub_debug_before_y -> publish(y_before);

  z_before.data = odom_point.position.z;
  pub_debug_before_z -> publish(z_before);

  double rune_angle = target_msg->angle;
  if (tracker_->target_status == TargetStatus::TRACK)
  {
    if ((static_cast<rclcpp::Time>(target_msg->header.stamp) - tracker_->stamp).seconds() > 0.07)
    {
      //reset
      RCLCPP_INFO(
        this->get_logger(), "Remove a lost AT: %lf",
        (static_cast<rclcpp::Time>(target_msg->header.stamp) - tracker_->stamp).seconds());
      tracker_->target_status = TargetStatus::LOST;
      tracker_->ekf.reset(new cv_model::EKF());
      tracker_->ekf->setP(cv_model::EKF::MatrixSS::Identity() * 1e5);
      tracker_->last_rune_angle = 0;
    }
    else if (std::fabs(tracker_->last_rune_angle - rune_angle) > deg2rad(15))
    {
      angle_err_count_++;
      if (angle_err_count_ > 5)
      {
        //reset
        tracker_->target_status = TargetStatus::LOST;
        tracker_->ekf.reset(new cv_model::EKF());
        tracker_->ekf->setP(cv_model::EKF::MatrixSS::Identity() * 1e5);
        tracker_->last_rune_angle = 0;
        angle_god_count_ = 0;
        angle_err_count_ = 0;
      }
      else
      {
        return;
      }
    }
    else
    {
      angle_god_count_++;
      if (angle_god_count_ > 10)
      {
        angle_god_count_ = 0;
        angle_err_count_ = 0;
      }
    }
  }

  //init filter
  if (tracker_->target_status == TargetStatus::LOST)
  {
    //reset
    cv_model::EKF::State x0;
    x0 << odom_point.position.x, 0, odom_point.position.y, 0, odom_point.position.z, 0;

    if (!tracker_->ekf)
    {
      tracker_->ekf = std::make_shared<cv_model::EKF>(x0);
    }
    else
    {
      tracker_->ekf.reset(new cv_model::EKF(x0));
    }
    tracker_->stamp = target_msg->header.stamp;
    tracker_->target_status = TargetStatus::TRACK;
    tracker_->last_rune_angle = rune_angle;
  }

  tracker_->last_rune_angle = rune_angle;
  // predict
  {
    const auto sampling_time =
      (static_cast<rclcpp::Time>(target_msg->header.stamp) - tracker_->stamp)
        .to_chrono<std::chrono::microseconds>();
    tracker_->stamp = target_msg->header.stamp;
    const std::chrono::microseconds dt = sampling_time;
    tracker_->ekf->predict(dt);
  }

  // update
  {
    const Eigen::Quaterniond camera2odom_quaterniond = {(*camera2odom_transform).transform.rotation.w,
                                                        (*camera2odom_transform).transform.rotation.x,
                                                        (*camera2odom_transform).transform.rotation.y,
                                                        (*camera2odom_transform).transform.rotation.z};
    tracker_->ekf->update(
      {odom_point.position.x, odom_point.position.y, odom_point.position.z},
      {camera_point.position.x, camera_point.position.y, camera_point.position.z},
      camera2odom_quaterniond.toRotationMatrix());
    tracker_->target_status = TargetStatus::TRACK;
  }
  if (debug_)
  {
    std_msgs::msg::Float64 msg;
    msg.data = tracker_->ekf->get_Chi();
    pub_debug_chi_->publish(msg);
  }
  // normal check
  {
    if (!tracker_->ekf->isnormal())
    {
      RCLCPP_WARN(this->get_logger(), "ekf is not normal, reset.");
      tracker_->target_status = TargetStatus::LOST;
      tracker_->ekf.reset(new cv_model::EKF());
    }
  }

  // publish
  {
    Eigen::Matrix<double, 6, 1> target_state = tracker_->ekf->getS_e();
    autoaim_sys_interfaces::msg::TrackedRune tracked_rune_msg;
    tracked_rune_msg.header.stamp = target_msg->header.stamp;
    tracked_rune_msg.header.frame_id = "odom_frame";
    tracked_rune_msg.is_tracked = (tracker_->target_status == TargetStatus::LOST) ? false : true;
    tracked_rune_msg.target_state.x = target_state[cv_model::EKF::X];
    tracked_rune_msg.target_state.y = target_state[cv_model::EKF::Y];
    tracked_rune_msg.target_state.z = target_state[cv_model::EKF::Z];
    tracked_rune_msg.target_state.dx = target_state[cv_model::EKF::dX];
    tracked_rune_msg.target_state.dy = target_state[cv_model::EKF::dY];
    tracked_rune_msg.target_state.dz = target_state[cv_model::EKF::dZ];

    std_msgs::msg::Float64 x, y, z;
    x.data = target_state[cv_model::EKF::X];
    y.data = target_state[cv_model::EKF::Y];
    z.data = target_state[cv_model::EKF::Z];
    pub_debug_x -> publish(x);
    pub_debug_y -> publish(y);
    pub_debug_z -> publish(z);
    pub_tracked_target_->publish(tracked_rune_msg);
  }

  // debug
  geometry_msgs::msg::PoseStamped debug_odom_target;
  debug_odom_target.header.stamp = target_msg->header.stamp;
  debug_odom_target.header.frame_id = "odom_frame";
  debug_odom_target.pose = odom_point;
  pub_odom_target_debug_->publish(debug_odom_target);
}

void TrackerNode::declareParams()
{
  RCLCPP_INFO(this->get_logger(), "start declare params");
  params_->sigma_s_x = this->declare_parameter<double>("sigma_s_x", 0.0);
  params_->sigma_s_y = this->declare_parameter<double>("sigma_s_y", 0.0);
  params_->sigma_s_z = this->declare_parameter<double>("sigma_s_z", 0.0);
  params_->sigma_m_x = this->declare_parameter<double>("sigma_m_x", 0.0);
  params_->sigma_m_alpha = this->declare_parameter<double>("sigma_m_alpha", 0.0);
  params_->sigma_m_beta = this->declare_parameter<double>("sigma_m_beta", 0.0);
  params_->chi_square_threshold = this->declare_parameter<double>("chi_square_threshold", 0.0);
}

void TrackerNode::updateParams()
{
  params_->sigma_s_x = this->get_parameter("sigma_s_x").as_double();
  params_->sigma_s_y = this->get_parameter("sigma_s_y").as_double();
  params_->sigma_s_z = this->get_parameter("sigma_s_z").as_double();
  params_->sigma_m_x = this->get_parameter("sigma_m_x").as_double();
  params_->sigma_m_alpha = this->get_parameter("sigma_m_alpha").as_double();
  params_->sigma_m_beta = this->get_parameter("sigma_m_beta").as_double();
  params_->chi_square_threshold = this->get_parameter("chi_square_threshold").as_double();

  tracker_->ekf->updateParams(
    params_->sigma_s_x, params_->sigma_s_y, params_->sigma_s_z, params_->sigma_m_x,
    params_->sigma_m_alpha, params_->sigma_m_beta, params_->chi_square_threshold);
}

}  //namespace rune

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rune::TrackerNode)