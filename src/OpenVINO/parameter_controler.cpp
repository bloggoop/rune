#include "OpenVINO/OpenVINO_node.hpp"

namespace rune
{
  void OpenVINO_node::declareParams(){
    try
    {
      rune_params_->RuneRadius = this->declare_parameter<double>("RuneRadius", rune_params_->RuneRadius);
      rune_params_->RHeight  = this->declare_parameter<double>("RHeight", rune_params_->RHeight);
      rune_status_str_ = this->declare_parameter<std::string>("autoaim_mode", "normal");
      RCLCPP_INFO(this->get_logger(), "OpenVINO declareParams success");
    }
    catch(const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "OpenVINO declareParams error: %s", e.what());
    }
  }

  void OpenVINO_node::updateParams(){
    try
    {
      rune_params_->RHeight = this->get_parameter("RHeight").as_double();
      rune_params_->RuneRadius = this->get_parameter("RuneRadius").as_double();
      rune_status_str_ = this->get_parameter("autoaim_mode").as_string();
      // RCLCPP_INFO(this->get_logger(), "OpenVINO updateParams success");
    }
    catch(const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(),"OpenVINO updateParams error: %s", e.what());
    }
  }
}




