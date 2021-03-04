##Build with: sudo docker build -t image_seg -f ./container/Dockerfile .

FROM ros:melodic-ros-core-bionic

LABEL maintainer="Katerina Oikonomou <koikonomou@iit.demokritos.gr>"

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
    xvfb  \
    ros-melodic-tf \
    ros-melodic-pcl-ros \
    ros-melodic-cv-bridge \
    ros-melodic-sensor-msgs \
    ros-melodic-message-filters \
    ros-melodic-laser-geometry \
    ros-melodic-laser-filters \
    ros-melodic-message-runtime \
    ros-melodic-pcl-conversions \
    ros-melodic-image-transport \
    ros-melodic-message-generation && \
    rm -rf /var/lib/apt-get/lists/*

RUN mkdir -p /opt/catkin_ws/src

RUN apt-get update && apt-get install -y \
		build-essential \
		git \
		python-rosdep && \
	rosdep init && \
	rosdep update && \
	cd /opt/catkin_ws/src && \
	git clone https://github.com/roboskel/pointcloud_msgs.git && \
	git clone https://github.com/roboskel/laserscan_stacker.git  && \
	git clone https://github.com/roboskel/pointcloud2_clustering.git && \
	git clone https://github.com/roboskel/pointcloud2_cluster_tracking.git && \
	git clone https://github.com/roboskel/pointcloud2_segments_viz.git && \
	git clone -b rel3 https://github.com/roboskel/hpr.git && \
	git clone https://github.com/roboskel/image_segmentation_node.git && \
	git clone https://github.com/roboskel/image_msgs.git && \
	git clone https://github.com/roboskel/roboskel_msgs.git && \
    cd .. && \
    rosdep install --from-paths src --ignore-src -r -y && \
    /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make --pkg roboskel_msgs" && \
    /bin/bash -c "source /opt/ros/melodic/setup.bash; source /opt/catkin_ws/devel/setup.bash; catkin_make" && \
    apt-get purge -y \
        build-essential \
        git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt-get/lists/*

RUN sed --in-place --expression \
    '$isource "/opt/catkin_ws/devel/setup.bash"' \
    /ros_entrypoint.sh

EXPOSE 80