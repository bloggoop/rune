#include "OpenVINO/solvepose.hpp"

namespace rune
{
SolvePose::SolvePose(const std::string config_filename)
{
  std::string filepath =
    ament_index_cpp::get_package_share_directory("rune") + "/config/" + config_filename;
  cv::FileStorage fs(filepath, cv::FileStorage::READ);

  if (!fs.isOpened())
  {
    throw std::runtime_error("Solvepose Error: filestorage failed to open TAML file.");
  }

  /* 如果直接以Mat类型读取，务必注意yaml文件格式要求 */
  cv::Mat dist_coeffs;
  fs["camera_matrix"] >> camera_matrix_;
  fs["distortion_coefficients"] >> dist_coeffs;

  for (int i = 0; i < dist_coeffs.cols; i++)
  {
    dist_coeffs_.emplace_back(dist_coeffs.at<double>(i));
  }

  fs.release();
}

SolvePose::SolvePose(const cv::Mat & camera_matrix, const std::vector<double> & dist_coeffs) :
camera_matrix_(camera_matrix), dist_coeffs_(dist_coeffs)
{
}

SolvePose::SolvePose(const std::array<double, 9> & camera_matrix,
                     const std::vector<double> & dist_coeffs) :
camera_matrix_(cv::Mat(camera_matrix)), dist_coeffs_(dist_coeffs)
{
}

void SolvePose::update(const cv::Mat & camera_matrix, const std::vector<double> & dist_coeffs)
{
  if (camera_matrix.empty()) return;
  if (dist_coeffs.empty()) return;

  camera_matrix_ = camera_matrix;
  dist_coeffs_   = dist_coeffs;
}

// int SolvePose::jugle_points(std::vector<cv::Point3f> result) {

// }

void SolvePose::solvePnP(const std::array<cv::Point3f, 4> & object_points,
  const std::array<cv::Point2f, 4> & image_points,cv::Mat & r_vec_,cv::Mat & t_vec_,cv::Mat& img_rec)
{
    if(camera_matrix_.empty() || dist_coeffs_.empty()) {
      std::throw_with_nested(std::runtime_error("Solvepose Error: camera_matrix or dist_coeffs is empty."));
    }
    cv::solvePnP(object_points, image_points, camera_matrix_, dist_coeffs_, r_vec_, t_vec_, false, cv::SOLVEPNP_IPPE);
    if(r_vec_.empty() || t_vec_.empty()) {
      std::throw_with_nested(std::runtime_error("Solvepose Error: solvePnP failed."));
  }
  // 画出image_points
  // for (size_t i = 0; i < image_points.size(); i++)
  // {
  //   cv::circle(img_rec, image_points[i], 5, cv::Scalar(0, 255, 0), -1);
  //   //标注对应的3维坐标系点
  //   cv::putText(img_rec,
  //               std::to_string(i),
  //               image_points[i],
  //               cv::FONT_HERSHEY_SIMPLEX,
  //               0.5,
  //               cv::Scalar(0, 255, 0),
  //               1);
  //   // std::cout<<object_points[i]<<std::endl;
  // }
}  
}// namespace rune