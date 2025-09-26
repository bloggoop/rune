#include "prediction/plane_solve.hpp"


namespace rune {
  RuneOdomSolver::RuneOdomSolver(int num_datas) : num_datas_(num_datas) {
    plane_ = std::make_unique<Plane>();
  }

  void RuneOdomSolver::updatePlane(Eigen::Vector3f basic_vector, Eigen::Vector3f normal_vector, Eigen::Vector3f plane_point) { 

    plane_->plane_basic_vector_ = point2plane(basic_vector).normalized();
    plane_->plane_point_ = plane_point;
    plane_->normal_vector_ = normal_vector;
    plane_->plane_params[0] = normal_vector[0];
    plane_->plane_params[1] = normal_vector[1];
    plane_->plane_params[2] = normal_vector[2];
    plane_->plane_params[3] = -normal_vector.dot(plane_point);
  }

  Eigen::Vector3f RuneOdomSolver::point2plane(Eigen::Vector3f point) {
    if(!is_plane_solved_){
      exit;
    }
    float a = point[0];
    float b = point[1];
    float c = point[2];

    float A = plane_ -> normal_vector_[0];
    float B = plane_ -> normal_vector_[1];
    float C = plane_ -> normal_vector_[2];
    float D = -A * plane_ -> plane_point_[0] - B * plane_ -> plane_point_[1] - C * plane_ -> plane_point_[2];
    float square_ABC = A * A + B * B + C * C;

    float x_0 = -A*(B*b + C*c + D) + a*(B*B + C*C) / square_ABC;
    float y_0 = -B*(A*a + C*c + D) + b*(A*A + C*C) / square_ABC;
    float z_0 = -C*(A*a + B*b + D) + c*(A*A + B*B) / square_ABC;

    return Eigen::Vector3f(x_0, y_0, z_0);
  }

  float RuneOdomSolver::solveAngle(Eigen::Vector3f vec_) {
    Eigen::Vector3f vec_plane = point2plane(vec_);
    Eigen::Vector3f vec_normal = vec_plane.normalized();
    Eigen::Vector3f vec_cross = vec_normal.cross(plane_ ->plane_basic_vector_).normalized();

    float vec_dot = vec_normal.dot(plane_ ->plane_basic_vector_);
    float angle = 0;

    if((vec_cross + plane_ -> normal_vector_).norm() > 1 ){
      angle = std::acos(vec_dot)+M_PI;
    }
    else{
      angle = M_PI-std::acos(vec_dot);
    }

    return angle;
  }

  Eigen::Vector3f RuneOdomSolver::Rodrigues(Eigen::Vector3f vec_, float angle_, Eigen::Vector3f normal_vector_) {
    Eigen::Vector3f k = normal_vector_.normalized();

    Eigen::Vector3f vec_n = -k.cross(k.cross(vec_));
    Eigen::Vector3f vec_p = k.dot(vec_) * k;

    Eigen::Vector3f vec_r_n = vec_n *cos(angle_) + k.cross(vec_n) * sin(angle_);

    return vec_r_n + vec_p;
  }
}




