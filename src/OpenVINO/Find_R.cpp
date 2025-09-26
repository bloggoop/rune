#include "OpenVINO/Find_R.hpp"

namespace rune{
  void RDetecter::setTarget(cv::Point2f target_center)
  {
    target_center_ = target_center; 
  }

  void RDetecter::draw_WaterfallLight_Rs(cv::Mat& img, const RDetecter::WaterfallLight_Rs& waterfall_lights) {
    for (const auto& waterfall_light : waterfall_lights)
    {
      cv::rectangle(img, waterfall_light.Rect_, cv::Scalar(0, 255, 0), 2);
      cv::circle(img, waterfall_light.center_, 5, cv::Scalar(0, 0, 255), -1);
    }
  }

  void RDetecter::preprocess(cv::Mat& img) {

    //直接二值化
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::threshold(img, *img_bin_, 150 , 255, cv::THRESH_BINARY);

    if (!contours_->empty())
    {
      contours_->clear();
      contours_->shrink_to_fit();
    }
    if (!hierarchies_->empty())
    {
      hierarchies_->clear();
      hierarchies_->shrink_to_fit();
    }
    cv::findContours(*img_bin_, *contours_, *hierarchies_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  }

  void RDetecter::findWaterfallLightAndR() {
    if(!waterfall_light_and_Rs_->empty()) {
      waterfall_light_and_Rs_->clear();
      waterfall_light_and_Rs_->shrink_to_fit();
    }

    for (size_t i = 0; i < contours_->size(); i++)
    {
      /* 寻找没有父轮廓和子轮廓的轮廓 */
      if ((*hierarchies_)[i][3] != -1 || (*hierarchies_)[i][2] != -1)
      {
        continue;
      }

      /* 面积排除噪声*/
      const double area = cv::contourArea(contours_->at(i));
      if (area < waterfall_light_params_->WaterfallLightMinArea ||
          area > waterfall_light_params_->WaterfallLightMaxArea)
      {
        // drawWrongContours(debug_params_->show_err_contours_,
        //                   *img_rec_,
        //                   contours_->at(i),
        //                   contours_->at(i).at(0),
        //                   "area",
        //                   area);
        continue;
      }

      /* 外接矩形宽高比排除噪声 */
      const cv::Rect boundingRect = cv::boundingRect(contours_->at(i));
      const double aspect_ratio   = static_cast<double>(std::min(boundingRect.width, boundingRect.height)) /
                                    static_cast<double>(std::max(boundingRect.width, boundingRect.height));
      if (aspect_ratio < waterfall_light_params_->WaterfallLightMinAspectRatio ||
          aspect_ratio > waterfall_light_params_->WaterfallLightMaxAspectRatio ||
          !std::isnormal(aspect_ratio))
      {
        // drawWrongContours(debug_params_->show_err_contours_,
        //                   *img_rec_,
        //                   contours_->at(i),
        //                   contours_->at(i).at(0),
        //                   "asp",
        //                   aspect_ratio);
        continue;
      }

      /* 填充率排除噪声 */
      std::vector<cv::Point> hull;
      cv::convexHull(contours_->at(i), hull);
      const double hull_area = cv::contourArea(hull);
      const double rof = area / hull_area;
      if (rof < waterfall_light_params_->WaterfallLightMinROF ||
          rof > waterfall_light_params_->WaterfallLightMaxROF ||
          !std::isnormal(rof))
      {
        // drawWrongContours(debug_params_->show_err_contours_,
        //                   *img_rec_,
        //                   contours_->at(i),
        //                   contours_->at(i).at(0),
        //                   "rof",
        //                   rof);
        continue;
      }
    
      //距离筛选
      cv::Point2f center = cv::Point2f(boundingRect.x , boundingRect.y);
      float dist_2_target = std::sqrt(std::pow(center.x - target_center_.x, 2) +
                                    std::pow(center.y - target_center_.y, 2));
      if(dist_2_target < waterfall_light_params_->dist_2_target_min ||
         dist_2_target > waterfall_light_params_->dist_2_target_max)
      {
        continue;
      }

      //存入容器
      waterfall_light_and_Rs_->emplace_back(
        dist_2_target,
        Eigen::Vector2f(boundingRect.x-target_center_.x, boundingRect.y-target_center_.y),
        boundingRect, 
        cv::Point2f(boundingRect.x, boundingRect.y));
    }

      //方向筛选
      for(auto  vector : *waterfall_light_and_Rs_)
      {
        target_vec_ = target_vec_ + vector.target_vector_;
      }
      for(auto vector : *waterfall_light_and_Rs_){
      }

       // 流水灯筛选
      if (waterfall_light_and_Rs_->size() > 3)
      {
        // 等距性检验
        std::unordered_set<int> distance_points{};
        const int n_light = waterfall_light_and_Rs_->size();
        for (int i = 0; i < n_light; i++)
        {
          for (int j = i + 1; j < n_light; j++)
          {
            for (int k = j + 1; k < n_light; k++)
            {
              const double dist_i_j = dist2D<cv::Point2f>(waterfall_light_and_Rs_->at(i).center_,
                                                         waterfall_light_and_Rs_->at(j).center_);
              const double dist_i_k = dist2D<cv::Point2f>(waterfall_light_and_Rs_->at(i).center_,
                                                         waterfall_light_and_Rs_->at(k).center_);
              const double dist_j_k = dist2D<cv::Point2f>(waterfall_light_and_Rs_->at(j).center_,
                                                         waterfall_light_and_Rs_->at(k).center_);
              std::vector<double> dist{dist_i_j, dist_i_k, dist_j_k};
              std::erase(dist, std::ranges::max(dist));
              if (std::abs(dist[0] - dist[1]) <= waterfall_light_params_->WaterfallLightMaxDistanceDiff)
              {
                distance_points.insert(i);
                distance_points.insert(j);
                distance_points.insert(k);
              }
            }
          }
        }

        // 共线性检验
        std::unordered_set<int> collinear_points{};
        std::vector<int> distance_points_idx(distance_points.begin(), distance_points.end());
        const int n_dist_points = distance_points_idx.size();
        for (int i = 0; i < n_dist_points; i++)
        {
          for (int j = i + 1; j < n_dist_points; j++)
          {
            for (int k = j + 1; k < n_dist_points; k++)
            {
              const auto & p1 = waterfall_light_and_Rs_->at(distance_points_idx.at(i)).center_;
              const auto & p2 = waterfall_light_and_Rs_->at(distance_points_idx.at(j)).center_;
              const auto & p3 = waterfall_light_and_Rs_->at(distance_points_idx.at(k)).center_;
              if (is_collinear(p1, p2, p3))
              {
                collinear_points.insert(i);
                collinear_points.insert(j);
                collinear_points.insert(k);
              }
            }
          }
        }

        // 构造最大共线等距集
        WaterfallLight_Rs tmp_storage{};
        std::for_each(collinear_points.begin(),
                      collinear_points.end(),
                      [&tmp_storage, this](const int index) {
                        tmp_storage.emplace_back(waterfall_light_and_Rs_->at(index));
                      });
        waterfall_light_and_Rs_->clear();
        waterfall_light_and_Rs_->shrink_to_fit();
        std::copy(tmp_storage.begin(), tmp_storage.end(), std::back_inserter(*waterfall_light_and_Rs_));
      }
  }

  void RDetecter::FindR (cv::Mat& img){
    if(img.empty()) {
      throw std::runtime_error("Input image is empty");
    }
    preprocess(img);
    findWaterfallLightAndR();
    if(waterfall_light_and_Rs_->empty()) {
      throw std::runtime_error("No waterfall light detected");
    }
    auto R_ptr = std::max_element(waterfall_light_and_Rs_->begin(), waterfall_light_and_Rs_->end(), 
                                            [](const WaterfallLightAndR& a, 
                                              const WaterfallLightAndR& b) 
                                            {
                                              return a.dist_2_target_ < b.dist_2_target_;
                                            });
    if (R_ptr == waterfall_light_and_Rs_->end()) {
      throw std::runtime_error("No R found!");
    }
    *R_ = *R_ptr;
  }
}



