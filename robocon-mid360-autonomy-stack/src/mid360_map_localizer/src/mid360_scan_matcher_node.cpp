#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

namespace
{

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Quaternionf q(
    static_cast<float>(pose.orientation.w),
    static_cast<float>(pose.orientation.x),
    static_cast<float>(pose.orientation.y),
    static_cast<float>(pose.orientation.z));
  if (q.norm() < 1e-6f) {
    q = Eigen::Quaternionf::Identity();
  } else {
    q.normalize();
  }
  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result.block<3, 3>(0, 0) = q.toRotationMatrix();
  result(0, 3) = static_cast<float>(pose.position.x);
  result(1, 3) = static_cast<float>(pose.position.y);
  result(2, 3) = static_cast<float>(pose.position.z);
  return result;
}

geometry_msgs::msg::Pose matrixToPose(const Eigen::Matrix4f & matrix)
{
  geometry_msgs::msg::Pose pose;
  const Eigen::Quaternionf q(matrix.block<3, 3>(0, 0));
  pose.position.x = matrix(0, 3);
  pose.position.y = matrix(1, 3);
  pose.position.z = matrix(2, 3);
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

double yawFromMatrix(const Eigen::Matrix4f & matrix)
{
  return std::atan2(static_cast<double>(matrix(1, 0)), static_cast<double>(matrix(0, 0)));
}

double wrapAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

Cloud::Ptr voxelDownsample(const Cloud::ConstPtr & input, const float leaf)
{
  auto output = std::make_shared<Cloud>();
  if (leaf <= 0.0f || input->empty()) {
    *output = *input;
    return output;
  }
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(input);
  filter.setLeafSize(leaf, leaf, leaf);
  filter.filter(*output);
  return output;
}

}  // namespace

class Mid360ScanMatcher : public rclcpp::Node
{
public:
  Mid360ScanMatcher()
  : Node("mid360_scan_matcher")
  {
    map_file_ = declare_parameter<std::string>("map_file", "");
    min_map_points_ = declare_parameter<int>("min_map_points", 500);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/mid360/cloud_registered_reliable");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/mid360/local_odometry");
    initialpose_topic_ = declare_parameter<std::string>("initialpose_topic", "/initialpose");
    correction_topic_ = declare_parameter<std::string>("correction_topic", "/mid360/map_to_odom_correction");
    diagnostic_topic_ = declare_parameter<std::string>("diagnostic_topic", "/mid360/map_localization_diagnostics");
    expected_scan_frame_ = declare_parameter<std::string>("expected_scan_frame", "");
    require_scan_frame_ = declare_parameter<bool>("require_scan_frame", false);
    min_scan_points_ = declare_parameter<int>("min_scan_points", 80);
    max_scan_points_ = declare_parameter<int>("max_scan_points", 30000);
    voxel_leaf_size_ = declare_parameter<double>("voxel_leaf_size", 0.12);
    map_voxel_leaf_size_ = declare_parameter<double>("map_voxel_leaf_size", 0.12);
    max_correspondence_distance_ = declare_parameter<double>("max_correspondence_distance", 1.5);
    max_iterations_ = declare_parameter<int>("max_iterations", 30);
    transformation_epsilon_ = declare_parameter<double>("transformation_epsilon", 0.0001);
    euclidean_fitness_epsilon_ = declare_parameter<double>("euclidean_fitness_epsilon", 0.001);
    max_fitness_score_ = declare_parameter<double>("max_fitness_score", 0.40);
    max_translation_jump_m_ = declare_parameter<double>("max_translation_jump_m", 0.80);
    max_yaw_jump_deg_ = declare_parameter<double>("max_yaw_jump_deg", 25.0);
    min_odom_freshness_sec_ = declare_parameter<double>("min_odom_freshness_sec", 0.30);
    publish_correction_hz_ = declare_parameter<double>("publish_correction_hz", 10.0);

    if (!map_file_.empty()) {
      loadMap(map_file_);
    } else {
      RCLCPP_WARN(get_logger(), "map_file is empty; fixed-map matching is disabled.");
      status_ = "WAITING_FOR_MAP";
    }

    correction_pub_ = create_publisher<nav_msgs::msg::Odometry>(correction_topic_, rclcpp::QoS(5));
    diagnostic_pub_ = create_publisher<std_msgs::msg::String>(diagnostic_topic_, rclcpp::QoS(5));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(20),
      std::bind(&Mid360ScanMatcher::odomCallback, this, std::placeholders::_1));
    initialpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      initialpose_topic_, rclcpp::QoS(10),
      std::bind(&Mid360ScanMatcher::initialposeCallback, this, std::placeholders::_1));
    scan_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      scan_topic_, rclcpp::QoS(5),
      std::bind(&Mid360ScanMatcher::scanCallback, this, std::placeholders::_1));

    if (publish_correction_hz_ <= 0.0) {
      throw std::runtime_error("publish_correction_hz must be positive");
    }
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_correction_hz_)),
      std::bind(&Mid360ScanMatcher::publishDiagnostic, this));
  }

private:
  void loadMap(const std::string & path)
  {
    auto raw = std::make_shared<Cloud>();
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *raw) != 0 || raw->empty()) {
      status_ = "MAP_LOAD_FAILED";
      RCLCPP_ERROR(get_logger(), "Could not load a non-empty PCD map: %s", path.c_str());
      return;
    }
    map_cloud_ = voxelDownsample(raw, static_cast<float>(map_voxel_leaf_size_));
    if (min_map_points_ <= 0) {
      status_ = "INVALID_MIN_MAP_POINTS";
      RCLCPP_ERROR(get_logger(), "min_map_points must be positive, got %d.", min_map_points_);
      return;
    }
    if (static_cast<int>(map_cloud_->size()) < min_map_points_) {
      status_ = "MAP_TOO_SMALL";
      RCLCPP_ERROR(
        get_logger(), "Rejected map %s: %zu points after downsampling, minimum is %d.",
        path.c_str(), map_cloud_->size(), min_map_points_);
      map_cloud_->clear();
      return;
    }
    map_loaded_ = true;
    status_ = map_loaded_ ? "WAITING_FOR_INITIALPOSE" : "MAP_LOAD_FAILED";
    RCLCPP_INFO(get_logger(), "Loaded fixed map %s with %zu points after downsampling.", path.c_str(), map_cloud_->size());
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr message)
  {
    if (message->header.frame_id != odom_frame_ || message->child_frame_id != base_frame_) {
      status_ = "REJECTED_ODOM_FRAME";
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odom_ = *message;
    latest_odom_arrival_ = now();
    trySetAnchorLocked();
  }

  void initialposeCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr message)
  {
    if (!message->header.frame_id.empty() && message->header.frame_id != map_frame_) {
      status_ = "REJECTED_INITIALPOSE_FRAME";
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    pending_initialpose_ = *message;
    trySetAnchorLocked();
  }

  void trySetAnchorLocked()
  {
    if (!pending_initialpose_.has_value() || !latest_odom_.has_value()) {
      return;
    }
    const double age = (now() - latest_odom_arrival_).seconds();
    if (age > min_odom_freshness_sec_) {
      status_ = "WAITING_FOR_FRESH_ODOM";
      return;
    }
    const Eigen::Matrix4f map_to_base = poseToMatrix(pending_initialpose_->pose.pose);
    const Eigen::Matrix4f odom_to_base = poseToMatrix(latest_odom_->pose.pose);
    map_to_odom_ = map_to_base * odom_to_base.inverse();
    anchor_set_ = true;
    pending_initialpose_.reset();
    status_ = map_loaded_ ? "TRACKING" : "WAITING_FOR_MAP";
    RCLCPP_INFO(get_logger(), "Accepted /initialpose and initialized map-to-odom anchor.");
  }

  Cloud::Ptr prepareScan(const sensor_msgs::msg::PointCloud2 & message) const
  {
    if (require_scan_frame_ && message.header.frame_id != expected_scan_frame_) {
      return std::make_shared<Cloud>();
    }
    auto raw = std::make_shared<Cloud>();
    pcl::fromROSMsg(message, *raw);
    auto finite = std::make_shared<Cloud>();
    finite->reserve(raw->size());
    for (const auto & point : raw->points) {
      if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
        finite->push_back(point);
      }
    }
    auto output = voxelDownsample(finite, static_cast<float>(voxel_leaf_size_));
    if (max_scan_points_ > 0 && static_cast<int>(output->size()) > max_scan_points_) {
      auto capped = std::make_shared<Cloud>();
      capped->reserve(static_cast<std::size_t>(max_scan_points_));
      const std::size_t stride = static_cast<std::size_t>(
        std::ceil(static_cast<double>(output->size()) / max_scan_points_));
      for (std::size_t i = 0; i < output->size() && capped->size() < static_cast<std::size_t>(max_scan_points_); i += stride) {
        capped->push_back(output->at(i));
      }
      return capped;
    }
    return output;
  }

  void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    if (!map_loaded_) {
      status_ = "WAITING_FOR_MAP";
      return;
    }
    Eigen::Matrix4f initial_guess;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!anchor_set_ || !latest_odom_.has_value()) {
        status_ = "WAITING_FOR_INITIALPOSE";
        return;
      }
      if ((now() - latest_odom_arrival_).seconds() > min_odom_freshness_sec_) {
        status_ = "STALE_ODOM";
        return;
      }
      initial_guess = map_to_odom_;
    }

    const auto scan = prepareScan(*message);
    if (static_cast<int>(scan->size()) < min_scan_points_) {
      status_ = "REJECTED_TOO_FEW_POINTS";
      return;
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(scan);
    icp.setInputTarget(map_cloud_);
    icp.setMaxCorrespondenceDistance(max_correspondence_distance_);
    icp.setMaximumIterations(max_iterations_);
    icp.setTransformationEpsilon(transformation_epsilon_);
    icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
    Cloud aligned;
    icp.align(aligned, initial_guess);
    if (!icp.hasConverged()) {
      status_ = "REJECTED_NOT_CONVERGED";
      return;
    }
    const double fitness = icp.getFitnessScore();
    if (!std::isfinite(fitness) || fitness > max_fitness_score_) {
      status_ = "REJECTED_FITNESS";
      return;
    }

    const Eigen::Matrix4f candidate = icp.getFinalTransformation();
    Eigen::Matrix4f previous_anchor;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      previous_anchor = map_to_odom_;
    }
    const Eigen::Matrix4f delta = previous_anchor.inverse() * candidate;
    const double translation_jump = delta.block<3, 1>(0, 3).norm();
    const double yaw_jump = std::abs(wrapAngle(yawFromMatrix(delta)));
    if (max_translation_jump_m_ > 0.0 && translation_jump > max_translation_jump_m_) {
      status_ = "REJECTED_TRANSLATION_JUMP";
      return;
    }
    if (max_yaw_jump_deg_ > 0.0 && yaw_jump > max_yaw_jump_deg_ * M_PI / 180.0) {
      status_ = "REJECTED_YAW_JUMP";
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      map_to_odom_ = candidate;
      last_fitness_ = fitness;
      last_scan_points_ = scan->size();
      status_ = "TRACKING";
    }
    nav_msgs::msg::Odometry correction;
    correction.header.stamp = message->header.stamp;
    correction.header.frame_id = map_frame_;
    correction.child_frame_id = odom_frame_;
    correction.pose.pose = matrixToPose(candidate);
    correction_pub_->publish(correction);
  }

  void publishDiagnostic()
  {
    std_msgs::msg::String message;
    std::lock_guard<std::mutex> lock(mutex_);
    // JSON has no NaN/Inf literal. Keep the diagnostic contract parseable
    // before the first accepted ICP result by emitting null for unknown data.
    const std::string fitness_json = std::isfinite(last_fitness_)
      ? std::to_string(last_fitness_) : "null";
    message.data =
      "{\"version\":1,\"status\":\"" + status_ +
      "\",\"map_loaded\":" + (map_loaded_ ? "true" : "false") +
      ",\"anchor_set\":" + (anchor_set_ ? "true" : "false") +
      ",\"last_fitness\":" + fitness_json +
      ",\"last_scan_points\":" + std::to_string(last_scan_points_) + "}";
    diagnostic_pub_->publish(message);
  }

  std::string map_file_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string scan_topic_;
  std::string odom_topic_;
  std::string initialpose_topic_;
  std::string correction_topic_;
  std::string diagnostic_topic_;
  std::string expected_scan_frame_;
  bool require_scan_frame_{false};
  int min_scan_points_{80};
  int min_map_points_{500};
  int max_scan_points_{30000};
  double voxel_leaf_size_{0.12};
  double map_voxel_leaf_size_{0.12};
  double max_correspondence_distance_{1.5};
  int max_iterations_{30};
  double transformation_epsilon_{0.0001};
  double euclidean_fitness_epsilon_{0.001};
  double max_fitness_score_{0.40};
  double max_translation_jump_m_{0.80};
  double max_yaw_jump_deg_{25.0};
  double min_odom_freshness_sec_{0.30};
  double publish_correction_hz_{10.0};

  Cloud::Ptr map_cloud_{std::make_shared<Cloud>()};
  bool map_loaded_{false};
  bool anchor_set_{false};
  Eigen::Matrix4f map_to_odom_{Eigen::Matrix4f::Identity()};
  std::optional<nav_msgs::msg::Odometry> latest_odom_;
  std::optional<geometry_msgs::msg::PoseWithCovarianceStamped> pending_initialpose_;
  rclcpp::Time latest_odom_arrival_{0, 0, RCL_ROS_TIME};
  double last_fitness_{std::numeric_limits<double>::quiet_NaN()};
  std::size_t last_scan_points_{0};
  std::string status_{"STARTING"};
  std::mutex mutex_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr correction_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostic_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Mid360ScanMatcher>());
  rclcpp::shutdown();
  return 0;
}
