#include "transform/transform_node.hpp"
#include <rclcpp/node_options.hpp>

namespace rune
{

TransformNode::TransformNode(const rclcpp::NodeOptions & options) :
rclcpp::Node("TransformNode", options)
{
  pub_fanblade_odom_ =
    this->create_publisher<rune_sys_interfaces::msg::Fanblade>(pub_fanblade_odom_topic_, 10);
  pub_camera2odom_ =
    this->create_publisher<geometry_msgs::msg::TransformStamped>(pub_camera2odom_topic_, 10);

  tf2_buffer_          = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
  std::chrono::duration<int> buffer_timeout(10);
  fanblade_sub_.subscribe(this, sub_fanblade_topic_name_);
  tf2_fanblade_filter_ =
    std::make_shared<tf2_ros::MessageFilter<rune_sys_interfaces::msg::Fanblade>>(
      fanblade_sub_,
      *tf2_buffer_,
      "odom_frame",
      100,
      this->get_node_logging_interface(),
      this->get_node_clock_interface(),
      buffer_timeout);

  tf2_fanblade_filter_->registerCallback(&TransformNode::fanbladeCallback, this);
}

TransformNode::~TransformNode() {}

void TransformNode::fanbladeCallback(const rune_sys_interfaces::msg::Fanblade::ConstSharedPtr & msg)
{
  /* -------------------------------------------- check ------------------------------------------- */
  if (msg->t_vec.empty())
  {
    RCLCPP_WARN(this->get_logger(), "fanblade-msg t_vec is empty!");
    return;
  }
  if (!std::isnormal(msg->quaternion.orientation.w) ||
      !std::isnormal(msg->quaternion.orientation.x) ||
      !std::isnormal(msg->quaternion.orientation.y) ||
      !std::isnormal(msg->quaternion.orientation.z))
  {
    RCLCPP_WARN(this->get_logger(), "fanblade-msg quaternion isnt normal.");
    return;
  }

  // geometry_msgs::msg::TransformStamped t;
  // t.header.stamp = (rclcpp::Time)msg->header.stamp +
  //     rclcpp::Duration::from_seconds(msg->reco_timecost * 1.0e-3);

  /* ---------------------------------- get camera2odom transform --------------------------------- */
  geometry_msgs::msg::TransformStamped camera2odom_transform;
  try
  {
    camera2odom_transform =
      tf2_buffer_->lookupTransform("odom_frame", "camera_frame", msg->header.stamp);
  } catch (const tf2::TransformException & ex)
  {
    RCLCPP_WARN(this->get_logger(), "TF2 Transform Failure: %s.", ex.what());
  } catch (...)
  {
    RCLCPP_ERROR(this->get_logger(), "TF2 Transform Error.");
  }
  pub_camera2odom_->publish(camera2odom_transform);

  /* ---------------------------------- transform fanblade to odom --------------------------------- */
  const double fanblade_center_cam[3] = {msg->fanblade_center.x,
                                      msg->fanblade_center.y,
                                      msg->fanblade_center.z};

  double  fanblade_center_odom[3];
  frameTransform2Odom(fanblade_center_cam, fanblade_center_odom, "camera_frame", msg->header.stamp);

  rune_sys_interfaces::msg::Fanblade fanblade_msg;
  fanblade_msg.header.stamp               = msg->header.stamp;
  fanblade_msg.quaternion                 = msg->quaternion;
  fanblade_msg.t_vec                      = msg->t_vec;
  fanblade_msg.fanblade_center.x          = fanblade_center_odom[0];
  fanblade_msg.fanblade_center.y          = fanblade_center_odom[1];
  fanblade_msg.fanblade_center.z          = fanblade_center_odom[2];
  fanblade_msg.now_angle                  = msg->now_angle;
  fanblade_msg.fanblade_angle_0           = msg->fanblade_angle_0;
  fanblade_msg.fanblade_now_id            = msg->fanblade_now_id;
  fanblade_msg.rotation                   = msg->rotation;
  pub_fanblade_odom_->publish(fanblade_msg);
}

void TransformNode::frameTransform2Odom(const double (&ori_pose)[3],
                                        double (&target_pose)[3],
                                        const std::string_view ori_frame,
                                        const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseStamped ori_posestamp;
  geometry_msgs::msg::PoseStamped target_posestamp;
  ori_posestamp.header.stamp       = stamp;
  ori_posestamp.header.frame_id    = ori_frame;
  ori_posestamp.pose.position.x    = ori_pose[0];
  ori_posestamp.pose.position.y    = ori_pose[1];
  ori_posestamp.pose.position.z    = ori_pose[2];
  target_posestamp.header.stamp    = stamp;
  target_posestamp.header.frame_id = "odom_frame";

  try
  {
    tf2_buffer_->transform(ori_posestamp, target_posestamp, "odom_frame");
  } catch (const tf2::TransformException & ex)
  {
    RCLCPP_WARN(this->get_logger(), "TF2 Transform Failure: %s.", ex.what());
  } catch (...)
  {
    RCLCPP_ERROR(this->get_logger(), "TF2 Transform Error.");
  }

  target_pose[0] = target_posestamp.pose.position.x;
  target_pose[1] = target_posestamp.pose.position.y;
  target_pose[2] = target_posestamp.pose.position.z;
}

}  // namespace rune

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rune::TransformNode)