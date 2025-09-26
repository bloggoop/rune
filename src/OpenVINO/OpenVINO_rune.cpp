#include<algorithm>
#include "OpenVINO/OpenVINO_rune.hpp"

namespace rune {

  OpenVINO_rune::OpenVINO_rune(std::string red_model_path, std::string blue_model_path, bool isRed, cv::Size target_size, float iou_threshlod)
    : red_model_path_(red_model_path), blue_model_path_(blue_model_path), isRed_(isRed), target_size_(target_size), iou_threshlod_(iou_threshlod)
  {
    ov::Core core;
    if (isRed) {
      model_ = core.read_model(red_model_path_);
    }
    else {
      model_ = core.read_model(blue_model_path_);
    }
    compiled_model_ = std::make_shared<ov::CompiledModel>(core.compile_model(model_, "CPU"));
    infer_request_ = std::make_shared<ov::InferRequest>(compiled_model_->create_infer_request());
  }

  OpenVINO_rune::OpenVINO_rune(std::string model_path, cv::Size target_size, float conf_threshlod, float iou_threshlod)
    : model_path_(model_path), target_size_(target_size), conf_threshlod_(conf_threshlod), iou_threshlod_(iou_threshlod)
  {
    num_keypoints_ = NUM_KEYPOINTS;
    ov::Core core;
    model_ = core.read_model(model_path);
    compiled_model_ = std::make_shared<ov::CompiledModel>(core.compile_model(model_, "CPU"));
    infer_request_ = std::make_shared<ov::InferRequest>(compiled_model_->create_infer_request());
  }
  OpenVINO_rune::~OpenVINO_rune() = default;

  void OpenVINO_rune::letterBox(cv::Mat& img) {
    if(img.empty()) {
        // std::cout<<"empty image!"<<std::endl;
        return;
    }
    int img_w = img.cols;
    int img_h = img.rows;
    int target_w = target_size_.width;
    int target_h = target_size_.height;
    float scale = std::min(float(target_w) / img_w, float(target_h) / img_h);
    int resize_w = int(img_w * scale);
    int resize_h = int(img_h * scale);
    int pad_w = (target_w - resize_w);
    int pad_h = (target_h - resize_h);
    int pad_w_r = (target_w - resize_w) - pad_w;
    int pad_h_b = (target_h - resize_h) - pad_h;

    transform_matrix_ << 
      1.0 / scale, 0, -pad_w / scale,
      0, 1.0 / scale, -pad_h / scale,
      0, 0, 1;

    cv::resize(img, img, cv::Size(resize_w, resize_h));
    cv::copyMakeBorder(img, img, pad_h, pad_h_b, pad_w, pad_w_r, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  }

  bool OpenVINO_rune::generateCandidates(std::array<std::vector<PosePoints>, NUM_CLASS>& points_info, const cv::Mat& output_buffer, float box_threshlod) {
    // std::cout << "enter generateCandidates()" << std::endl;
    for (int i = 0; i < output_buffer.cols; i++) {
      std::vector<float> confidences;
      for(int  j = 0; j< NUM_CLASS; j++) {
        confidences.emplace_back(output_buffer.at<float>(4 + j, i));
      }
      // 置信度阈值过滤
      std::vector<float>::iterator max_confidence = std::max_element(confidences.begin(), confidences.end());
      if(*max_confidence < box_threshlod) {
        continue;
      }
      int class_num = std::distance(confidences.begin(), max_confidence);

      PosePoints points;
      // 存入关键点信息和置信度到PosePoints结构体
      points.prob = *max_confidence;
      std::vector<cv::Point2f> keypoints;
      for (int j = 0; j < NUM_KEYPOINTS; j++) {
        float x = output_buffer.at<float>(4 + NUM_CLASS + j * 2, i);
        float y = output_buffer.at<float>(4 + NUM_CLASS + j * 2 + 1, i);
        points.keypoints.emplace_back(cv::Point2f(x, y));
      }

      float up_left_x = output_buffer.at<float>(0, i) - output_buffer.at<float>(2, i)/2;
      float up_left_y = output_buffer.at<float>(1, i) - output_buffer.at<float>(3, i)/2;
      cv::Point2f up_left(up_left_x, up_left_y);
      float down_right_x = output_buffer.at<float>(0, i) + output_buffer.at<float>(2, i)/2;
      float down_right_y = output_buffer.at<float>(1, i) + output_buffer.at<float>(3, i)/2;
      cv::Point2f down_right(down_right_x, down_right_y);
      cv::Rect2f rect(up_left, down_right);
      // std::cout << "up_left_x: " << up_left_x << " up_left_y: " << up_left_y << " down_right_x: " << down_right_x << " down_right_y: " << down_right_y << std::endl;

      points.box = rect;
      points.pose_class = class_num;

      points_info[class_num].emplace_back(points);

      // for (int j = 0; j < output_buffer.rows; j++) {
      //   std::cout << output_buffer.at<float>(j, i) << " ";
      // }
      // std::cout <<"class_num: "<<points.pose_class <<std::endl;
    }

    if(points_info.size() == 0) return false;
    else return true;
  }

  float OpenVINO_rune::calculate_iou(cv::Rect a, cv::Rect b) {
    cv::Rect intersection = a & b;
    if (intersection.area() == 0) {
      return 0;
    }
    cv::Rect union_set = a | b;
    return float(intersection.area()) / float(union_set.area());
  }

  std::array<std::vector<PosePoints>, NUM_CLASS> OpenVINO_rune::nms(std::array<std::vector<PosePoints>, NUM_CLASS>& points_info) {
    if(points_info[NetClass::TARGET].size() == 0) {
      return {};
    }
    std::array<std::vector<PosePoints>, NUM_CLASS> result;
  
    for(int i = 0; i < NUM_CLASS; i++) {
        std::sort(points_info[i].begin(), points_info[i].end(), 
                  [](const PosePoints& a, const PosePoints& b) {
                    return a.prob > b.prob;
                  });
        
        //如果是TARGET类别，直接返回第一个元素就行
        if(i == NetClass::TARGET) {
          result[i].push_back(points_info[i][0]);
          continue;
        }
        
        std::vector<PosePoints> class_result;  

        for(int j = 0; j < points_info[i].size(); j++) {
            bool keep = true;  

            for(int k = 0; k < class_result.size(); k++) {
                if(calculate_iou(class_result[k].box, points_info[i][j].box) > iou_threshlod_) {
                    keep = false; 
                    break;
                }
            }
            if(keep) {
                class_result.push_back(points_info[i][j]);
            }
        }
        result[i].insert(result[i].end(), class_result.begin(), class_result.end());
    }

    //处理两个类别互相重叠的情况，保留靶标
    for(int i = 0; i < result.size(); i++) {
      for(int j = 0; j < result[i].size(); j++) {
        if(result[i][j].pose_class == NetClass::TARGET) {
          for(int k = j+1; k < result[i].size(); k++) {
            double iou = calculate_iou(result[i][j].box, result[i][k].box);
            if(iou > iou_threshlod_) {
              if(result[i][k].prob > result[i][j].prob) {
                result[i][j] = result[i][k];
              }
              result[i].erase(result[i].begin() + k);
              k--;
            }
          }
        }
      }
      // if(result[i].pose_class == static_cast<int>(NetClass::TARGET)) {
      //   for(int j = i+1; j < result.size(); j++) {
      //     double iou = calculate_iou(result[i].box, result[j].box);
      //     if(iou > iou_threshlod_) {
      //       result.erase(result.begin() + j);
      //       break;
      //     }
      //   }
      // }
    }

    return result;
  }

  void OpenVINO_rune::transform_points_letbox_2_realimg(std::vector<cv::Point2f>& points_info) {
    const int size = points_info.size();
    float x[size];
    float y[size];
    for (int i = 0; i < size; i++) {
      x[i] = points_info[i].x;
      y[i] = points_info[i].y;
    }

    Eigen::Matrix <float, 3, Eigen::Dynamic> letterbox_points_matrix (3, size);
    for( int i = 0; i < size; i++) {
        letterbox_points_matrix(0, i) = x[i];
        letterbox_points_matrix(1, i) = y[i];
        letterbox_points_matrix(2, i) = 1;
    }

    Eigen::Matrix <float, 3, Eigen::Dynamic> real_matrix = transform_matrix_ * letterbox_points_matrix;

    for (int i = 0; i < size; i++) {
      points_info[i].x = real_matrix(0, i);
      points_info[i].y = real_matrix(1, i);
    }
  }
  void OpenVINO_rune::transform_points_letbox_2_realimg(cv::Rect &rect) {
    double x[4];
    double y[4];
    x[0] = rect.x;
    y[0] = rect.y;
    x[1] = rect.x + rect.width;
    y[1] = rect.y;
    x[2] = rect.x + rect.width;
    y[2] = rect.y + rect.height;
    x[3] = rect.x;
    y[3] = rect.y + rect.height;

    Eigen::Matrix <float, 3, Eigen::Dynamic> letterbox_points_matrix (3, 4);
    for( int i = 0; i < 4; i++) {
        letterbox_points_matrix(0, i) = x[i];
        letterbox_points_matrix(1, i) = y[i];
        letterbox_points_matrix(2, i) = 1;
    }

    Eigen::Matrix <float, 3, Eigen::Dynamic> real_matrix = transform_matrix_ * letterbox_points_matrix;

    for (int i = 0; i < 4; i++) {
      x[i] = real_matrix(0, i);
      y[i] = real_matrix(1, i);
    }
    cv::Point up_left(x[0], y[0]);
    cv::Point down_right(x[2], y[2]);
    rect = cv::Rect(up_left, down_right);
  }

  void OpenVINO_rune::optimizePoints(std::vector<cv::Point2f>& points, cv::Mat& img_bin) {
    if(points.empty() || img_bin.empty()) {
      throw std::runtime_error("empty points or img_bin!");
      return;
    }

    cv::cornerSubPix(
      img_bin,
      points,
      cv::Size(3, 3),
      cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1)
    );
  }

  void OpenVINO_rune::preprocess(cv::Mat& img, ov::element::Type type) {
    letterBox(img);
    //用cv::dnn::blobFromImage顺手将将图像转化为RGB格式
    cv::dnn::blobFromImage(img, blob_, 1.0 / 255.0f, target_size_, cv::Scalar(0, 0, 0), true, false);
    //将输入数据复制到输入张量,yolov8的输入尺寸为640x640x3
    input_tensor_ = ov::Tensor(type, { 1, 3, 640, 640 }, blob_.ptr<float>());
  }

  bool OpenVINO_rune::infer() {
    ov::InferRequest infer_request = compiled_model_->create_infer_request();
    infer_request.set_input_tensor(input_tensor_);
    infer_request.infer();

    ov::Tensor output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    output_buffer_ = cv::Mat(output_shape[1], output_shape[2], CV_32FC1, output_tensor.data());

    // 判断是否为空
    if(output_buffer_.empty()){
      return false;
    }
    return true;
  }

  std::array<std::vector<PosePoints>, NUM_CLASS> OpenVINO_rune::postprocess() {
    std::array<std::vector<PosePoints>, NUM_CLASS> points_info;
    if(!generateCandidates(points_info, output_buffer_, conf_threshlod_)) {
      return {};
    }
    // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "generateCandidates end");

    std::array<std::vector<PosePoints>, NUM_CLASS> result = nms(points_info);
    // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "%d, %d", result[0].size(), result[1].size());
    for(auto& class_point : result){
      for (auto& pinfo : class_point) {
        transform_points_letbox_2_realimg(pinfo.keypoints);
        transform_points_letbox_2_realimg(pinfo.box);
      }
    }
    
    return result;
  }

  std::array<std::vector<PosePoints>, NUM_CLASS> OpenVINO_rune::detect(cv::Mat img) {
    
    if(img.empty()) {
        // std::cout<<"empty image!"<<std::endl;
        return {};
    }
    // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "detect start");
    preprocess(img, ov::element::f32);
    // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "preprocess end");
    bool infer_success = infer();
    // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "infer end");
    if(infer_success){
      // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "postprocess start");
      return postprocess();
    }
    else {
      throw std::runtime_error("OpenVINO inference failed!");
      return {};
    }
  }

  RuneRotationStatue OpenVINO_rune::judgeRotation(Eigen::Vector2f now_vec,Eigen::Vector2f last_vec) {
      if(last_vec_== Eigen::Vector2f(0,0) || now_vec_ == Eigen::Vector2f(0,0) ) return RuneRotationStatue::NONE;
      float cross_product = CrossMultplication(now_vec, last_vec);
      if(cross_product > 0){
        return RuneRotationStatue::CW;
      }else return RuneRotationStatue::CCW;
    return RuneRotationStatue::NONE;
  }

  // bool OpenVINO_rune::checkContinuous(double angle) {
  //   if(now_angles_.size() < 3) {
  //     now_angles_.push_back(angle);
  //     return false; // 不足三个角度，无法判断连续性
  //   }
  //   now_angles_.pop_front();
  //   now_angles_.push_back(angle);

  //   double angle1 = now_angles_[now_angles_.size() - 1];
  //   double angle2 = now_angles_[now_angles_.size() - 2];
  //   double angle3 = now_angles_[now_angles_.size() - 3];

  //   // 判断三个角度是否连续
  //   double angle_diff1;
  //   double angle_diff2;
  //   if(rune_rotation_  == RuneRotationStatue::CW) {
  //     angle_diff1 = angle2 - angle1;
  //     angle_diff2 = angle3 - angle2;
  //   } 
  //   else if(rune_rotation_ == RuneRotationStatue::CCW) {
  //     angle_diff1 = angle1 - angle2;
  //     angle_diff2 = angle2 - angle3;
  //   } 
  //   else {
  //     return false;
  //   }
  //   // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "angle_diff1 = %f, angle_diff2 = %f", angle_diff1, angle_diff2);
  //   normAngle(angle_diff1);
  //   normAngle(angle_diff2);
  //   // RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "after norm,angle_diff1 = %f, angle_diff2 = %f", angle_diff1, angle_diff2);
  //   if(angle_diff1 < 0.3) {
  //     RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "angle_diff1 = %f, angle_diff2 = %f", angle_diff1, angle_diff2);
  //     return true;
  //   }
  //   return false;
  // }

  void OpenVINO_rune::updateAngle(double angle) {
  
    double now_angle = angle;

    if(last_angle_ == 0.0) {
      fanblade_angle_0_ = angle;
      fanblade_id_ = 0;
      last_angle_ = angle;
      // last_time_ = std::chrono::high_resolution_clock::now();
      return;
    }

    auto now_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - last_time_).count();
    if(std::abs(now_angle - last_angle_) / (duration *0.015) > 0.25 || std::abs(now_angle - last_angle_) > 0.35) {
      RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "jump angle, now_angle: %f, last_angle: %f, dif_angle: %f, duration: %d, threshold: %f, fanblade_angle_0: %f, fanblade_id: %d" , now_angle, last_angle_, std::abs(now_angle - last_angle_), duration, std::abs(now_angle - last_angle_) / (duration *0.015), fanblade_angle_0_, fanblade_id_);
      if(error_angle_count_ == 0) {
        last_angle_before_saltation_ = last_angle_;
      }
      error_angle_count_++;
      last_angle_ = now_angle;
      return;
    }
    else if (error_angle_count_ > 0 && std::abs(now_angle - last_angle_) < 0.35) {
      //代表已经切换到下一个叶片
      error_angle_count_ = 0;
      dif_angle_ = now_angle - last_angle_before_saltation_;

      normAngle(dif_angle_);
      RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "Switch id, now_angle: %f, last_angle: %f, dif_angle: %f, last_angle_before_saltation_: %f, fanblade_angle_0: %f, fanblade_id: %d", now_angle, last_angle_, dif_angle_, last_angle_before_saltation_, fanblade_angle_0_, fanblade_id_);
      
      int angle_remain = std::abs(static_cast<int>(dif_angle_ / M_PI*180) % 72);
    
      if(dif_angle_ > 0.9) {
        //判断切换的是哪个叶片
        int min_id_best = 0;
        int min_id_second = 0;
        double min_angle = 2 * M_PI;
        double min_angle_second = 2 * M_PI;
        for(int i = 0; i < 5; i++) {
          double angle_rotated = angle - i * 72.0 * M_PI / 180.0;
          normAngle(angle_rotated);
          double dif_angle_fanblade_0_abs = std::abs(angle_rotated - fanblade_angle_0_);
          double dif_angle_fanblade_norm  = angle_rotated - fanblade_angle_0_;
          normAngle(dif_angle_fanblade_norm);
          
          if(dif_angle_fanblade_0_abs < min_angle) {
            min_angle = dif_angle_fanblade_0_abs;
            min_id_second = min_id_best;
            min_id_best = i;
          }
          else if(dif_angle_fanblade_0_abs > min_angle && dif_angle_fanblade_0_abs < min_angle_second) {
            min_angle_second = dif_angle_fanblade_0_abs;
            min_id_second = i;
          }
          else if (dif_angle_fanblade_0_abs > 5.5 && dif_angle_fanblade_norm < min_angle) {
            min_angle = dif_angle_fanblade_0_abs;
            min_id_second = min_id_best;
            min_id_best = i;
          }
          else if (dif_angle_fanblade_0_abs > 5.5 && dif_angle_fanblade_norm > min_angle && dif_angle_fanblade_0_abs < min_angle_second)  {
            min_angle_second = dif_angle_fanblade_0_abs;
            min_id_second = min_id_best;
          }
        }

        double fanblade_angle_0_tmp = angle - min_id_best * 72.0 * M_PI / 180.0;
        normAngle(fanblade_angle_0_tmp);
        RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "min_id_best: %d, min_id_second: %d, fanblade_angle_0_tmp: %f, fanblade_angle_0_: %f", min_id_best, min_id_second, fanblade_angle_0_tmp, fanblade_angle_0_);
        rune_rotation_ = RuneRotationStatue::CCW;
        if(rune_rotation_ == RuneRotationStatue::CCW && fanblade_angle_0_tmp < fanblade_angle_0_) {
          double fanblade_angle_0_diff = fanblade_angle_0_tmp - fanblade_angle_0_;
          if(fanblade_angle_0_diff < 0) {
            RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "CCW,fanblade_angle_0_tmp: %f, fanblade_angle_0_: %f", fanblade_angle_0_tmp, fanblade_angle_0_);
            fanblade_id_ = min_id_best - 1;
          }
        }
        else if(rune_rotation_ == RuneRotationStatue::CW && fanblade_angle_0_tmp > fanblade_angle_0_) {
          double fanblade_angle_0_diff = fanblade_angle_0_tmp - fanblade_angle_0_;
          if(fanblade_angle_0_diff > 0) {
            RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "CW, fanblade_angle_0_tmp: %f, fanblade_angle_0_: %f", fanblade_angle_0_tmp, fanblade_angle_0_);
            fanblade_id_ = min_id_second + 1;
          }
        }
        else {
          fanblade_id_ = min_id_best;
        }

        RCLCPP_INFO(rclcpp::get_logger("OpenVINO_rune"), "after switch fanblade_id_ = %d", fanblade_id_);
      }
    }
    else {
      fanblade_angle_0_ = angle - 72.0 * M_PI / 180.0 * fanblade_id_;
      normAngle(fanblade_angle_0_);

      last_angle_ = now_angle;
      last_time_ = std::chrono::high_resolution_clock::now();
    }
  }
  void OpenVINO_rune::normAngle(double& angle) {
    angle = fmod(angle, 2* M_PI);
    if(angle < 0) angle += 2 * M_PI;
  }

  float OpenVINO_rune::CrossMultplication(Eigen::Vector2f v1,Eigen::Vector2f v2) {
    return v1.x()*v2.y() - v1.y()*v2.x();
  }

  cv::Mat OpenVINO_rune::draw_pose(cv::Mat& img, std::array<std::vector<PosePoints>, NUM_CLASS>& points_info) {
    // std::cout<<"enter draw_pose()"<<std::endl;
    for(auto tmp : points_info) {
      for (auto pinfo : tmp) {
        cv::rectangle(img, pinfo.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, std::to_string(pinfo.pose_class), pinfo.box.tl()+cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(100, 0, 255), 1);
        cv::putText(img, std::to_string(pinfo.prob), pinfo.box.tl()+cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(100, 0, 255), 1);
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
          cv::circle(img, pinfo.keypoints[i], 2, cv::Scalar(0, 255, 0), -1);
          cv::putText(img, std::to_string(i), pinfo.keypoints[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
      }
    }
    return img;
  }

  // int OpenVINO_rune::findMinAngleId() {
  //   float min_angle = 2*M_PI;
  //   int min_angle_id = 0;
  //   float angle_diff = 0;
  //   for(int i = 0; i < 1; i++) {
  //     angle_diff = std::abs(fanblade_angles_[i] - now_angle_);
  //     if(angle_diff < min_angle) {
  //       min_angle = angle_diff;
  //       min_angle_id = i;
  //     }
  //   }
  //   return min_angle_id;
  // }

}// namespace rune