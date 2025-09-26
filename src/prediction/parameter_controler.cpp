#include "prediction/prediction_node.hpp"

namespace rune
{
  void PredictionNode::declareParams(){
    try
    {
      // predict_params_->rune_radius_ = this-><double>("RuneRadius", predict_params_->RuneRadius);
      predict_params_->time_offset_  = this->declare_parameter<double>("time_offset", predict_params_->time_offset_);
      
      predict_params_->pre_angle_offset_ = this->declare_parameter<double>("PreAngleOffset", predict_params_->pre_angle_offset_);
      rune_bullet_params_. bullet_v = this->declare_parameter<double>("bullet_v", rune_bullet_params_.bullet_v);
      rune_bullet_params_. bullet_m = this->declare_parameter<double>("bullet_m", rune_bullet_params_.bullet_m);
      rune_bullet_params_. bullet_r = this->declare_parameter<double>("bullet_r", rune_bullet_params_.bullet_r);
      rune_bullet_params_. g        = this->declare_parameter<double>("g", rune_bullet_params_.g);
      rune_bullet_params_. rho      = this->declare_parameter<double>("rho", rune_bullet_params_  .rho);
      rune_bullet_params_. mu       = this->declare_parameter<double>("mu", rune_bullet_params_.mu);
      
      RCLCPP_INFO(this->get_logger(), "rune/PredictionNode declareParams success");
    }
    catch(const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "OpenVINO declareParams error: %s", e.what());
    }
  }

  void PredictionNode::updateParams(){
    try
    {
      // predict_params_->rune_radius_ = this->get_parameter("RuneRadius").as_double();
      predict_params_->time_offset_ = this->get_parameter("time_offset").as_double();
      predict_params_->pre_angle_offset_ = this->get_parameter("PreAngleOffset").as_double();

      rune_bullet_params_.bullet_v = this->get_parameter("bullet_v").as_double();
      rune_bullet_params_.bullet_m = this->get_parameter("bullet_m").as_double();
      rune_bullet_params_.bullet_r = this->get_parameter("bullet_r").as_double();
      rune_bullet_params_.g        = this->get_parameter("g").as_double();
      rune_bullet_params_.rho      = this->get_parameter("rho").as_double();
      rune_bullet_params_.mu       = this->get_parameter("mu").as_double();

      rune_bullet_params_.bullet_v = 23.0;
      rune_bullet_params_.bullet_m = 0.0032;
      rune_bullet_params_.bullet_r = 0.0085;
      rune_bullet_params_.g        = 9.8;
      rune_bullet_params_.rho     = 1.225;
      rune_bullet_params_.mu      = 0.26;
    }
    catch(const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(),"OpenVINO updateParams error: %s", e.what());
    }
  }
}




