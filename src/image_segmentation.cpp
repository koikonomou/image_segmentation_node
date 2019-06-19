#include <ros/ros.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <deque>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/point_cloud_conversion.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>

#include <pointcloud_msgs/PointCloud2_Segments.h>
#include <image_msgs/Image_Segments.h>

#define LOGFILE "imseg.log"
#define FIELD_OF_VIEW_ANGLE M_PI/3  //total camera field of view (horizontal) in rads
//#define MY_CLUSTER 3  //ID of desired cluster, for the purposes of the demo

using namespace std;
ros::Publisher pub;
image_transport::Publisher tpub;
int safety_pixels, maxBufferSize, MY_CLUSTER, cutPixelsFromTop, saveTotalPixels, clusterToChoose;

ros::Publisher pcl_pub;

sensor_msgs::Image latest_frame;
int first_frame = 0;
bool motioncluster = false;
double rad_deviation_Camera_Laser, base;


std::deque<sensor_msgs::Image> buffer;


void Log (uint32_t duration_nsecs, std::string message){  // logs a message to LOGFILE

  std::ofstream ofs;
  ofs.open(LOGFILE, std::ofstream::out | std::ios::app);
  ofs << duration_nsecs << " nsecs : " << message << std::endl;
  ofs.close();
}


void pcl_seg_Callback(const pointcloud_msgs::PointCloud2_Segments& msg) {

  if(first_frame == 0){
    return;
  }


  image_msgs::Image_Segments out_msg;
 

  double angle_min = msg.angle_min;
  double angle_max = msg.angle_max;
  double angle_increment = msg.angle_increment;

   
  ros::WallDuration dur;
  ros::WallDuration z_dur;
  uint32_t nsecs = 0;
  uint32_t z_nsecs = 0;

  //Timestamp: "Start" (Just got the pcl_segments message)
  ros::WallTime start = ros::WallTime::now();


  int pos = 0;

  for(unsigned b=buffer.size()-1; b>0 ; b--){

   if(msg.rec_time - buffer.at(b).header.stamp >= ros::Duration (0.0) ) {

      pos = b;

      break;
    }   
  }


  latest_frame = buffer.at(pos);
  buffer.erase(buffer.begin(), buffer.begin() + pos);


  if(clusterToChoose==2 && motioncluster==false && msg.idForTracking!=-1){
    motioncluster=true;
    MY_CLUSTER=msg.idForTracking;
  }


  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(latest_frame, "bgr8");


  pcl::PointCloud<pcl::PointXYZ> pcz;       //pcz contains all points with max z


  std::cout << "\n________NEW MESSAGE________\n" << std::endl;


  // Timestamp: "z_start_time" (Starting maximum z pointcloud extraction process)
  ros::WallTime z_start_time = ros::WallTime::now();


  // Timestamp: "z_stop_time" (End of maximum z pointcloud extraction process)
  ros::WallTime z_stop_time = ros::WallTime::now();
  z_dur = z_stop_time - z_start_time;
  z_nsecs = z_dur.toNSec();
  Log(z_nsecs,"Duration of Slice Extraction Process (all clusters)");

  for (int j=0; j < msg.clusters.size(); j++){    //for every cluster

    if(msg.cluster_id.size() != 0){
      if(msg.cluster_id[j] == MY_CLUSTER ){

        sensor_msgs::PointCloud2 pcl_out;
        pcl_out.header.frame_id = msg.header.frame_id;
        pcl_out.height = msg.clusters[j].height;
        pcl_out.width = msg.clusters[j].width;
        pcl_out.fields = msg.clusters[j].fields;
        pcl_out.is_bigendian = msg.clusters[j].is_bigendian;
        pcl_out.point_step = msg.clusters[j].point_step;
        pcl_out.row_step = msg.clusters[j].row_step;
        pcl_out.is_dense = msg.clusters[j].is_dense;
        pcl_out.data = msg.clusters[j].data;
        
        pcl_pub.publish(pcl_out);
      }
    }

    pcz.clear();

    double angle_l, angle_r, c_angle_l, c_angle_r;

    pcl::PCLPointCloud2 pc2;
    pcl_conversions::toPCL ( msg.clusters[j] , pc2 ); //from sensor_msgs::pointcloud2 to pcl::pointcloud2

    pcl::PointCloud<pcl::PointXYZ> pc;
    pcl::fromPCLPointCloud2 ( pc2 , pc );       //from pcl::pointcloud2 to pcl::pointcloud
    //pc is clusters[j] in pointcloud format

    

    double max_z = pc.points[0].z;
    for (int i=1; i < pc.points.size(); i++){   //find max z of cluster 
      if(pc.points[i].z > max_z){
        max_z = pc.points[i].z;
      }
    }

    for(int i=0; i < pc.points.size(); i++){    //add points with max z to a new pointcloud
      if(pc.points[i].z == max_z){
        pcz.push_back(pc.points[i]);
      }
    }

    pcl::PointXYZ min_point(pcz.points[0]);
    pcl::PointXYZ max_point(pcz.points[0]);

    double min_point_x = pcz.points[0].x;
    double max_point_x = pcz.points[0].x;


    for (int i=1; i < pcz.points.size(); i++){    //for every point in the cluster, find min y and max y  
      if(pcz.points[i].y < min_point.y){
        min_point.x = pcz.points[i].x;
        min_point.y = pcz.points[i].y;
        min_point.z = pcz.points[i].z;
      }
      if(pcz.points[i].y > max_point.y){
        max_point.x = pcz.points[i].x;
        max_point.y = pcz.points[i].y;
        max_point.z = pcz.points[i].z;
      }
      if(pcz.points[i].x < min_point_x){
        min_point_x = pcz.points[i].x;
      }
      if(pcz.points[i].x > max_point_x){
        max_point_x = pcz.points[i].x;
      }
    }

    angle_l = atan2(-min_point.y, min_point.x) + rad_deviation_Camera_Laser;
    angle_r = atan2(-max_point.y, max_point.x) + rad_deviation_Camera_Laser;

    int ratio = (cv_ptr->image.cols) / (FIELD_OF_VIEW_ANGLE) ;  //  (width pixels) / (field of view angle)
    int center = (cv_ptr->image.cols)/2;
    int x_l, x_r;


    //Pixel Conversion

    x_l = min(2*center, max(0, (int)ceil(min(angle_l, angle_r) * ratio + center - safety_pixels)));
    x_r = max(0, min(2*center, (int)floor(max(angle_l, angle_r) * ratio + center + safety_pixels)));

    int width_pixels, width_offset, height_pixels,height_offset;
    
    width_pixels = x_r - x_l;

    if (width_pixels < (1 + 2*safety_pixels)){
      out_msg.has_image.push_back(0);
    }
    else{
      out_msg.has_image.push_back(1);
      width_offset = x_l;


      //-----------------crop image horizontally -------------------------//

      double aver_dist = (min_point_x + max_point_x)/2; 

      height_offset = cutPixelsFromTop + pow(base, (int)ceil(aver_dist));
      height_pixels = saveTotalPixels - pow(base, (int)ceil(aver_dist)); 

      if(height_offset<0) height_offset = 0;
      else if(height_offset>cv_ptr->image.rows) {
        height_offset = cv_ptr->image.rows;
        height_pixels = 0;
      }

      if(height_pixels<=0) height_pixels = 0;
      else if(height_offset + height_pixels > cv_ptr->image.rows) height_pixels = cv_ptr->image.rows - height_offset;

      //----------------------------------------------------------------//

      cv::Rect myROIseg(width_offset, height_offset, width_pixels, height_pixels); 
      cv::Mat roiseg = cv::Mat(cv_ptr->image, myROIseg);

      //Image Segmentation

      sensor_msgs::ImagePtr imgptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", roiseg).toImageMsg();

      if(msg.cluster_id.size() != 0){
        if(msg.cluster_id[j] == MY_CLUSTER ){
          tpub.publish(*imgptr);
        }
      }
      
      out_msg.image_set.push_back(*imgptr); //insert images into message for publishing

      if(msg.cluster_id.size() != 0 ){
        std::cout << "cluster_id[" << j << "] : " << msg.cluster_id[j] << std::endl;
      }
      else{
        std::cout << "empty!!!!!!!!!!!!!!!" << std::endl;
      }
    }
  }

  std::cout << "\tSize of image set: " << out_msg.image_set.size() << std::endl;
  std::cout << "No of clusters: " << msg.clusters.size() << std::endl;
  
  std::cout << "cluster_id contents: [";
  if(msg.cluster_id.size()!=0){
   for(int c=0; c < msg.cluster_id.size(); c++){
    std::cout << "," << msg.cluster_id[c];
   }
    std::cout << "]" << std::endl;
  }

   std::cout << "has_image contents:  [";
  for(int c=0; c<out_msg.has_image.size(); c++){
   std::cout << "," << out_msg.has_image[c];
  }
  std::cout << "]" << std::endl;

  //Message Creation

  out_msg.header.stamp = ros::Time::now();
  out_msg.header.frame_id = msg.header.frame_id;
  out_msg.factor = msg.factor;
  out_msg.first_stamp = msg.first_stamp;
  out_msg.overlap = msg.overlap;
  out_msg.num_scans = msg.num_scans;
  out_msg.angle_min = msg.angle_min;
  out_msg.angle_max = msg.angle_max;
  out_msg.angle_increment = msg.angle_increment;
  out_msg.time_increment = msg.time_increment;
  out_msg.range_min = msg.range_min;
  out_msg.range_max = msg.range_max;
  out_msg.scan_time = msg.scan_time;
  out_msg.clusters = msg.clusters;
  out_msg.cluster_id = msg.cluster_id;

  // Timestamp: "End" (Just before publishing the message)
  ros::WallTime end = ros::WallTime::now();
  dur = end - start;
  nsecs = dur.toNSec();
  Log(nsecs,"Total Duration (from getting the message to just before publishing the new message)\n");

  pub.publish(out_msg);

}

void videoCallback(const sensor_msgs::ImageConstPtr& msg){

  std_msgs::Header h = msg->header;
  sensor_msgs::Image new_msg;

  new_msg = *msg;
  
  buffer.push_back(new_msg);

  if(buffer.size()>maxBufferSize) buffer.erase(buffer.begin(), buffer.begin() + buffer.size()-maxBufferSize);


  first_frame = 1;

  try{
    cv_bridge::CvImagePtr cv_ptr;
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


int main(int argc, char **argv){

  std::ofstream ofs;
  ofs.open(LOGFILE, std::ios::out | std::ios::trunc);   //clear log file
  ofs.close();

  ros::init(argc, argv, "image_segmentation_node");
  ros::NodeHandle nh;

  
  nh.param("image_segmentation_node/safety_pixels", safety_pixels, 19);
  nh.param("image_segmentation_node/maxBufferSize", maxBufferSize, 119);
  nh.param("image_segmentation_node/MY_CLUSTER", MY_CLUSTER, 3);
  nh.param("image_segmentation_node/cutPixelsFromTop", cutPixelsFromTop, 150);
  nh.param("image_segmentation_node/saveTotalPixels", saveTotalPixels, 60);
  nh.param("image_segmentation_node/rad_deviation_Camera_Laser", rad_deviation_Camera_Laser, 0.0);
  nh.param("image_segmentation_node/base", base, 2.0);
  nh.param("image_segmentation_node/clusterToChoose", clusterToChoose, 1);

  std::string input_topic_clusters;
  std::string out_topic_seg_images;
  std::string input_topic_camera;
  std::string out_topic_my_cluster_image;
  std::string out_topic_clusters;


  nh.param("image_segmentation_node/input_topic_clusters", input_topic_clusters, std::string("pointcloud2_cluster_tracking/clusters"));

  nh.param("image_segmentation_node/out_topic_seg_images", out_topic_seg_images, std::string("seg_images"));

  nh.param("image_segmentation_node/input_topic_camera", input_topic_camera, std::string("rear_cam/image_raw"));

  nh.param("image_segmentation_node/out_topic_my_cluster_image", out_topic_my_cluster_image, std::string("image_segmentation_node/seg_image"));

  nh.param("image_segmentation_node/out_topic_clusters", out_topic_clusters, std::string("test_pcl"));


  cout << "safety_pixels= " << safety_pixels << endl;
  cout << "maxBufferSize= " << maxBufferSize << endl;
  cout << "MY_CLUSTER= " << MY_CLUSTER << endl;
  cout << "base= " << base << endl;


  image_transport::ImageTransport it(nh);

  pub = nh.advertise<image_msgs::Image_Segments>(out_topic_seg_images, 2);
  
  tpub = it.advertise(out_topic_my_cluster_image, 1);

  pcl_pub = nh.advertise<sensor_msgs::PointCloud2> (out_topic_clusters, 1);

  ros::Subscriber pcl_seg_sub = nh.subscribe<const pointcloud_msgs::PointCloud2_Segments&>(input_topic_clusters, 1, pcl_seg_Callback);

  image_transport::Subscriber video_sub = it.subscribe(input_topic_camera, 50, videoCallback);    //  camera/rgb/image_raw gia to rosbag me tous 3, rear_cam/image_raw gia to rosbag me emena, usb_cam/image_raw gia to rosbag me to video mono

  ros::spin();
}
