##Build with: sudo docker build -t image_track -f ./container/Dockerfile .

FROM ros:noetic

LABEL maintainer="Katerina Oikonomou <koikonomou@iit.demokritos.gr>"

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
    xvfb  \
    ros-noetic-tf \
    ros-noetic-pcl-ros \
    ros-noetic-cv-bridge \
    ros-noetic-sensor-msgs \
    ros-noetic-message-filters \
    ros-noetic-laser-geometry \
    ros-noetic-laser-filters \
    ros-noetic-message-runtime \
    ros-noetic-pcl-conversions \
    ros-noetic-image-transport \
    ros-noetic-message-generation && \
    rm -rf /var/lib/apt-get/lists/*

RUN mkdir -p /opt/catkin_ws/src
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv

RUN python3 -m pip install \
    matplotlib \
    sklearn \
    numpy

RUN apt-get update && apt-get install -y \
		build-essential \
		git && \
	# 	python-rosdep && \
	# rosdep init && \
	# rosdep update && \
	cd /opt/catkin_ws/src && \
	git clone https://github.com/roboskel/pointcloud_msgs.git && \
	git clone https://github.com/roboskel/laserscan_stacker.git  && \
	git clone https://github.com/roboskel/pointcloud2_clustering.git && \
	git clone https://github.com/roboskel/pointcloud2_cluster_tracking.git && \
	git clone https://github.com/roboskel/pointcloud2_segments_viz.git && \
	git clone -b rel3 https://github.com/roboskel/hpr.git && \
	git clone https://github.com/koikonomou/image_segmentation_node.git && \
	git clone https://github.com/roboskel/image_msgs.git && \
	git clone https://github.com/koikonomou/roboskel_msgs.git && \
    cd .. && \
    rosdep install --from-paths src --ignore-src -r -y && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make --pkg roboskel_msgs" && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; source /opt/catkin_ws/devel/setup.bash; catkin_make" && \
    apt-get purge -y \
        build-essential \
        git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt-get/lists/*

RUN sed --in-place --expression \
    '$isource "/opt/catkin_ws/devel/setup.bash"' \
    /ros_entrypoint.sh

EXPOSE 80