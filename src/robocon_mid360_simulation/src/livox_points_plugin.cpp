#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/PhysicsIface.hh>
#include <gazebo/physics/MultiRayShape.hh>  // Store the latest laser scans into laserMsg
#include <gazebo/physics/PhysicsEngine.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <gazebo/transport/Node.hh>
#include <gazebo_ros/node.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <livox_ros_driver2/msg/custom_point.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "robocon_mid360_simulation/csv_reader.hpp"
#include "robocon_mid360_simulation/livox_points_plugin.h"
#include "robocon_mid360_simulation/livox_ode_multiray_shape.h"

namespace gazebo
{

    constexpr double kScanPatternTickToNanoseconds = 1000.0;

    double PatternOffsetNanoseconds(
        const double pattern_tick,
        const double frame_start_tick,
        const double period_ticks)
    {
        if (period_ticks <= 0.0) {
            return 0.0;
        }

        double relative_ticks = std::fmod(pattern_tick - frame_start_tick, period_ticks);
        if (relative_ticks < 0.0) {
            relative_ticks += period_ticks;
        }
        return relative_ticks * kScanPatternTickToNanoseconds;
    }

    GZ_REGISTER_SENSOR_PLUGIN(LivoxPointsPlugin)

    LivoxPointsPlugin::LivoxPointsPlugin() {}

    LivoxPointsPlugin::~LivoxPointsPlugin() {}

    void convertDataToRotateInfo(const std::vector<std::vector<double>> &datas, std::vector<AviaRotateInfo> &avia_infos)
    {
        avia_infos.reserve(datas.size());
        double deg_2_rad = M_PI / 180.0;
        for (auto &data : datas)
        {
            if (data.size() == 3)
            {
                avia_infos.emplace_back();
                avia_infos.back().time = data[0];
                avia_infos.back().azimuth = data[1] * deg_2_rad;
                avia_infos.back().zenith = data[2] * deg_2_rad - M_PI_2; //转化成标准的右手系角度
                ignition::math::Quaterniond ray;
                ray.Euler(ignition::math::Vector3d(
                    0.0, avia_infos.back().zenith, avia_infos.back().azimuth));
                avia_infos.back().direction =
                    ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            } else {
            RCLCPP_ERROR(rclcpp::get_logger("convertDataToRotateInfo"), "data size is not 3!");
        }
        }
    }

    void LivoxPointsPlugin::Load(gazebo::sensors::SensorPtr _parent, sdf::ElementPtr sdf)
    {
        node_ = gazebo_ros::Node::Get(sdf);
        
        std::vector<std::vector<double>> datas;
        std::string file_name = sdf->Get<std::string>("csv_file_name");
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "load csv file name: %s", file_name.c_str());
        if (!CsvReader::ReadCsvFile(file_name, datas))
        {   
            RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "cannot get csv file! %s will return !", file_name.c_str());
            return;
        }
        sdfPtr = sdf;
        auto rayElem = sdfPtr->GetElement("ray");
        auto scanElem = rayElem->GetElement("scan");
        auto rangeElem = rayElem->GetElement("range");


        raySensor = _parent;
        world = physics::get_world(raySensor->WorldName());
        if (!world)
        {
            RCLCPP_ERROR(rclcpp::get_logger("LivoxPointsPlugin"), "cannot resolve Gazebo world");
            return;
        }
        const auto update_rate = raySensor->UpdateRate();
        if (update_rate > 0.0)
        {
            scanPeriodSeconds = 1.0 / update_rate;
        }
        auto sensor_pose = raySensor->Pose();
        auto curr_scan_topic = sdf->Get<std::string>("topic");
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "ros topic name: %s", curr_scan_topic.c_str());

        child_name = raySensor->Name();
        parent_name = raySensor->ParentName();
        size_t delimiter_pos = parent_name.find("::");
        parent_name = parent_name.substr(delimiter_pos + 2);

        node = transport::NodePtr(new transport::Node());
        node->Init(raySensor->WorldName());
        // PointCloud2 publisher
        cloud2_pub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(curr_scan_topic + "/pointcloud", 10);
        // CustomMsg publisher
        custom_pub = node_->create_publisher<livox_ros_driver2::msg::CustomMsg>(curr_scan_topic, 10);

        scanPub = node->Advertise<msgs::LaserScanStamped>(curr_scan_topic+"laserscan", 50);

        aviaInfos.clear();
        convertDataToRotateInfo(datas, aviaInfos);
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "scan info size: %ld", aviaInfos.size());
        maxPointSize = aviaInfos.size();
        const auto tick_bounds = std::minmax_element(
            aviaInfos.begin(), aviaInfos.end(),
            [](const AviaRotateInfo &left, const AviaRotateInfo &right) {
                return left.time < right.time;
            });
        if (tick_bounds.first == aviaInfos.end() || tick_bounds.second == aviaInfos.end()) {
            RCLCPP_ERROR(rclcpp::get_logger("LivoxPointsPlugin"), "scan pattern is empty");
            return;
        }
        scanPatternFirstTick = tick_bounds.first->time;
        scanPatternPeriodTicks = tick_bounds.second->time - scanPatternFirstTick + 1.0;
        if (scanPatternPeriodTicks <= 0.0) {
            RCLCPP_ERROR(rclcpp::get_logger("LivoxPointsPlugin"), "invalid scan pattern tick range");
            return;
        }

        laserMsg.mutable_scan()->set_frame(_parent->ParentName());
        // parentEntity = world->GetEntity(_parent->ParentName());
        parentEntity = this->world->EntityByName(_parent->ParentName());
        //SendRosTf(sensor_pose, raySensor->ParentName(), raySensor->Name());
        auto physics = world->Physics();
        laserCollision = physics->CreateCollision("multiray", _parent->ParentName());
        laserCollision->SetName("ray_sensor_collision");
        laserCollision->SetRelativePose(_parent->Pose());
        laserCollision->SetInitialRelativePose(_parent->Pose());
        rayShape.reset(new gazebo::physics::LivoxOdeMultiRayShape(laserCollision));
        laserCollision->SetShape(rayShape);
        samplesStep = sdfPtr->Get<int>("samples");
        downSample = sdfPtr->Get<int>("downsample");
        if (downSample < 1)
        {
            downSample = 1;
        }
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "sample: %ld", samplesStep);
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "downsample: %ld", downSample);
        rayShape->RayShapes().reserve(samplesStep / downSample);
        rayShape->Load(sdfPtr);
        rayShape->Init();
        minDist = rangeElem->Get<double>("min");
        maxDist = rangeElem->Get<double>("max");
        auto offset = laserCollision->RelativePose();
        ignition::math::Vector3d start_point, end_point;
        for (int j = 0; j < samplesStep; j += downSample)
        {
            int index = j % maxPointSize;
            auto &rotate_info = aviaInfos[index];
            auto axis = offset.Rot() * rotate_info.direction;
            start_point = minDist * axis + offset.Pos();
            end_point = maxDist * axis + offset.Pos();
            rayShape->AddRay(start_point, end_point);
        }

        // Gazebo Classic may not emit RaySensor update callbacks in headless WSL.
        // Drive the same collision-backed callback from the world event instead.
        worldUpdateConnection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&LivoxPointsPlugin::OnWorldUpdate, this, std::placeholders::_1));
    }

    void LivoxPointsPlugin::OnWorldUpdate(const common::UpdateInfo &_info)
    {
        if (!raySensor || !rayShape)
        {
            return;
        }
        if (lastScanSimTime != common::Time::Zero &&
            (_info.simTime - lastScanSimTime).Double() + 1e-9 < scanPeriodSeconds)
        {
            return;
        }
        lastScanSimTime = _info.simTime;
        OnNewLaserScans();
    }



    void LivoxPointsPlugin::OnNewLaserScans() {
        if (!rayShape) {
            return; // 检查是否已经初始化了 rayShape
        }

        std::vector<std::pair<int, AviaRotateInfo>> points_pair;
        InitializeRays(points_pair, rayShape);
        rayShape->Update();

        msgs::Set(laserMsg.mutable_time(), world->SimTime());
        msgs::LaserScan *scan = laserMsg.mutable_scan();
        InitializeScan(scan);

        // Keep CustomMsg timing tied to Gazebo time, never wall-clock scheduling.
        livox_ros_driver2::msg::CustomMsg pp_livox;
        const double simulation_time_sec = world->SimTime().Double();
        const auto frame_seconds = static_cast<int32_t>(std::floor(simulation_time_sec));
        const auto frame_nanoseconds = static_cast<uint32_t>(
            std::llround((simulation_time_sec - static_cast<double>(frame_seconds)) * 1.0e9));
        pp_livox.header.stamp.sec = frame_seconds;
        pp_livox.header.stamp.nanosec = frame_nanoseconds;
        pp_livox.header.frame_id = raySensor->Name();
        pp_livox.timebase = static_cast<uint64_t>(frame_seconds) * 1000000000ULL + frame_nanoseconds;
        pp_livox.lidar_id = 0;

        // Build the PointCloud2 mirror only while a visualization consumer is
        // attached. FAST-LIO consumes CustomMsg, so avoiding an unused second
        // 30,000-point serialization does not alter the estimator input.
        const bool publish_cloud_mirror =
            cloud2_pub && cloud2_pub->get_subscription_count() > 0;
        sensor_msgs::msg::PointCloud2 cloud2;
        if (publish_cloud_mirror) {
            cloud2.header.stamp = node_->get_clock()->now();
            cloud2.header.frame_id = raySensor->Name();
        }

        struct TimedPoint {
            double pattern_tick;
            double x;
            double y;
            double z;
            float intensity;
        };
        std::vector<TimedPoint> valid_points;
        valid_points.reserve(points_pair.size());

        for (const auto &pair : points_pair) {
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);

            // Do not place no-return samples in the Livox packet.
            if (range <= RangeMin() || range >= RangeMax()) {
                continue;
            }

            const auto &rotate_info = pair.second;
            auto point = range * rotate_info.direction;

            valid_points.push_back({
                rotate_info.time, point.X(), point.Y(), point.Z(), static_cast<float>(intensity)});
        }

        // Ray order is the physical scan order. Numeric CSV ticks wrap after a full pattern,
        // so sorting by tick would incorrectly move the post-wrap points to the packet front.
        const double frame_start_pattern_tick = points_pair.empty() ? scanPatternFirstTick : points_pair.front().second.time;
        uint32_t previous_offset_ns = 0;
        pp_livox.points.reserve(valid_points.size());
        for (const auto &timed_point : valid_points) {
            // The upstream MID-360 CSV stores integer scan-pattern ticks in microseconds.
            // Use the complete CSV pattern period so frames that cross the tick wrap remain continuous.
            const double relative_ns = PatternOffsetNanoseconds(
                timed_point.pattern_tick, frame_start_pattern_tick, scanPatternPeriodTicks);
            const auto bounded_ns = std::min(
                relative_ns, static_cast<double>(std::numeric_limits<uint32_t>::max()));
            const uint32_t offset_ns = std::max(
                previous_offset_ns, static_cast<uint32_t>(std::llround(bounded_ns)));
            previous_offset_ns = offset_ns;

            livox_ros_driver2::msg::CustomPoint point;
            point.offset_time = offset_ns;
            point.x = static_cast<float>(timed_point.x);
            point.y = static_cast<float>(timed_point.y);
            point.z = static_cast<float>(timed_point.z);
            point.reflectivity = static_cast<uint8_t>(std::min(255.0F, std::max(0.0F, timed_point.intensity)));
            point.tag = 0;
            point.line = 0;
            pp_livox.points.push_back(point);
        }

        if (publish_cloud_mirror) {
            sensor_msgs::PointCloud2Modifier modifier(cloud2);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            modifier.resize(pp_livox.points.size());
            sensor_msgs::PointCloud2Iterator<float> out_x(cloud2, "x");
            sensor_msgs::PointCloud2Iterator<float> out_y(cloud2, "y");
            sensor_msgs::PointCloud2Iterator<float> out_z(cloud2, "z");
            for (const auto &point : pp_livox.points) {
                *out_x = point.x;
                *out_y = point.y;
                *out_z = point.z;
                ++out_x;
                ++out_y;
                ++out_z;
            }
        }

        if (scanPub && scanPub->HasConnections()) {
            scanPub->Publish(laserMsg);
        }

        pp_livox.point_num = static_cast<uint32_t>(pp_livox.points.size());
        custom_pub->publish(pp_livox);

        if (publish_cloud_mirror) {
            cloud2_pub->publish(cloud2);
        }
    }


    void LivoxPointsPlugin::InitializeRays(std::vector<std::pair<int, AviaRotateInfo>> &points_pair,
                                           boost::shared_ptr<physics::LivoxOdeMultiRayShape> &ray_shape)
    {
        auto &rays = ray_shape->RayShapes();
        ignition::math::Vector3d start_point, end_point;
        auto offset = laserCollision->RelativePose();
        int64_t end_index = currStartIndex + samplesStep;
        long unsigned int ray_index = 0;
        auto ray_size = rays.size();
        points_pair.reserve(rays.size());
        for (int k = currStartIndex; k < end_index; k += downSample)
        {
            auto index = k % maxPointSize;
            auto &rotate_info = aviaInfos[index];
            auto axis = offset.Rot() * rotate_info.direction;
            start_point = minDist * axis + offset.Pos();
            end_point = maxDist * axis + offset.Pos();
            if (ray_index < ray_size)
            {
                rays[ray_index]->SetPoints(start_point, end_point);
                points_pair.emplace_back(ray_index, rotate_info);
            }
            ray_index++;
        }
        currStartIndex += samplesStep;
    }

    void LivoxPointsPlugin::InitializeScan(msgs::LaserScan *&scan)
    {
        // Store the latest laser scans into laserMsg
        msgs::Set(scan->mutable_world_pose(), raySensor->Pose() + parentEntity->WorldPose());
        scan->set_angle_min(AngleMin().Radian());
        scan->set_angle_max(AngleMax().Radian());
        scan->set_angle_step(AngleResolution());
        scan->set_count(RangeCount());

        scan->set_vertical_angle_min(VerticalAngleMin().Radian());
        scan->set_vertical_angle_max(VerticalAngleMax().Radian());
        scan->set_vertical_angle_step(VerticalAngleResolution());
        scan->set_vertical_count(VerticalRangeCount());

        scan->set_range_min(RangeMin());
        scan->set_range_max(RangeMax());

        scan->clear_ranges();
        scan->clear_intensities();

        unsigned int rangeCount = RangeCount();
        unsigned int verticalRangeCount = VerticalRangeCount();

        for (unsigned int j = 0; j < verticalRangeCount; ++j)
        {
            for (unsigned int i = 0; i < rangeCount; ++i)
            {
                scan->add_ranges(0);
                scan->add_intensities(0);
            }
        }
    }

    ignition::math::Angle LivoxPointsPlugin::AngleMin() const
    {
        if (rayShape)
            return rayShape->MinAngle();
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::AngleMax() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->MaxAngle().Radian());
        }
        else
            return -1;
    }

    double LivoxPointsPlugin::GetRangeMin() const { return RangeMin(); }

    double LivoxPointsPlugin::RangeMin() const
    {
        if (rayShape)
            return rayShape->GetMinRange();
        else
            return -1;
    }

    double LivoxPointsPlugin::GetRangeMax() const { return RangeMax(); }

    double LivoxPointsPlugin::RangeMax() const
    {
        if (rayShape)
            return rayShape->GetMaxRange();
        else
            return -1;
    }

    double LivoxPointsPlugin::GetAngleResolution() const { return AngleResolution(); }

    double LivoxPointsPlugin::AngleResolution() const { return (AngleMax() - AngleMin()).Radian() / (RangeCount() - 1); }

    double LivoxPointsPlugin::GetRangeResolution() const { return RangeResolution(); }

    double LivoxPointsPlugin::RangeResolution() const
    {
        if (rayShape)
            return rayShape->GetResRange();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetRayCount() const { return RayCount(); }

    int LivoxPointsPlugin::RayCount() const
    {
        if (rayShape)
            return rayShape->GetSampleCount();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetRangeCount() const { return RangeCount(); }

    int LivoxPointsPlugin::RangeCount() const
    {
        if (rayShape)
            return rayShape->GetSampleCount() * rayShape->GetScanResolution();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetVerticalRayCount() const { return VerticalRayCount(); }

    int LivoxPointsPlugin::VerticalRayCount() const
    {
        if (rayShape)
            return rayShape->GetVerticalSampleCount();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetVerticalRangeCount() const { return VerticalRangeCount(); }

    int LivoxPointsPlugin::VerticalRangeCount() const
    {
        if (rayShape)
            return rayShape->GetVerticalSampleCount() * rayShape->GetVerticalScanResolution();
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::VerticalAngleMin() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->VerticalMinAngle().Radian());
        }
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::VerticalAngleMax() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->VerticalMaxAngle().Radian());
        }
        else
            return -1;
    }

    double LivoxPointsPlugin::GetVerticalAngleResolution() const { return VerticalAngleResolution(); }

    double LivoxPointsPlugin::VerticalAngleResolution() const
    {
        return (VerticalAngleMax() - VerticalAngleMin()).Radian() / (VerticalRangeCount() - 1);
    }


}
