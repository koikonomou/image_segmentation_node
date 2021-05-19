#!/usr/bin/env python3
import sys
import cv2
import rospy
import argparse
from sensor_msgs.msg import Image
from image_msgs.msg import Image_Segments
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import numpy as np

class Track:
    centers = None
    colors_ = [(int(color[2]), int(color[1]), int(color[0])) for color in [(np.array(color) * 255).astype(np.uint8) for name, color in mcolors.BASE_COLORS.items()]]
    # pool_ = multiprocessing.Pool(multiprocessing.cpu_count())

    def __init__(self):
      print("A")
      # self.image_pub = rospy.Publisher("image_topic_2",Image)

      self.bridge = CvBridge()
      self.image_sub = rospy.Subscriber("seg_images", Image_Segments, self.track_callback)
      self.num_clusters = 1


    def find_dist_(self, prev_center, new_center):
        print("4")
        """Find distance between two Lab color centers"""
        return 8 * np.sqrt((float(prev_center) - float(new_center)) ** 2 + (float(prev_center) - float(new_center)) ** 2) + 2 * abs(prev_center - new_center)
        # return 8 * np.sqrt((float(prev_center[1]) - float(new_center[1])) ** 2 + (float(prev_center[2]) - float(new_center[2])) ** 2) + 2 * abs(prev_center[0] - new_center[0])

    def find_all_dists_(self, centers):
        print("5")
        """Find all distances between 2 lists of color centers"""
        ret = []
        for cen_i, center in enumerate(centers):
            for c_i, c in enumerate(self.centers):
                ret.append((self.find_dist_(c, center), (cen_i, c_i)))
        return sorted(ret, key = lambda x: x[0])

    def update_all_dists_(self, all_dists, match):
        """Remove distances of centers that have already been matched"""
        print("6")
        return [i for i in all_dists if (i[1][0] != match[0] and i[1][1] != match[1])]

    def update_cur_center_(self, matches, center):
        """Update current center"""
        matched = []
        print("7")
        for match in matches:
            matched.append(match[0])
            self.centers[match[1]] = (center[match[0]] + self.centers[match[1]]) / 2
        for i in range(len(center)):
            if i not in matched:
                self.centers.append(center[i])

    def find_dom_color_(self,image_gb):
        """Find dominant color of upper body part using kmeans clustering"""
        print("8")
        img = cv2.cvtColor(image_gb, cv2.COLOR_BGR2LAB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        #kmeans to find dominant color


        clt = KMeans(n_clusters = self.num_clusters)
        clt.fit(img)
        centers_ = clt.cluster_centers_
        temp = np.unique(clt.labels_, return_counts = True)[1]
        center = list(map(lambda a: a[1], sorted(enumerate(centers_), key = lambda a: temp[a[0]], reverse = True)))[0]
        return center
        #Person mask is not eligible
        # return None

    # def fix_masks_(self, masks, indices):
    #     """Remove from every mask any common pixels with another mask"""
    #     new_masks = []
    #     for ind in indices:
    #         mask = masks[ind]
    #         for ind1 in indices:
    #             if ind != ind1:
    #                 mask = np.logical_and(mask, np.logical_and(masks[ind], np.logical_not(masks[ind1])))
    #         new_masks.append(mask)
    #     return new_masks

    def visualize_tracking_(self, img, indices, rect):
        """Show image with colored bounding boxes on people"""
        img1 = np.copy(img)
        print("9")
        print("len indices", len(indices))
        for i in range(len(indices)):
            print("rectangle", rect[i])
            img1 = cv2.rectangle(img1, rect[i], self.colors_[self.ids[i] % len(self.colors_)], 2)
        cv2.imshow("image", img1)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

    def track_callback(self, msg):
        print("1") 
        image_gb = msg.full_image
        image_gb = self.bridge.imgmsg_to_cv2(image_gb, "bgr8")
        image_gb = cv2.GaussianBlur(image_gb, (7, 7), 0)

        # print(1)
        #Gaussian Blurring
        # image_gb = cv2.GaussianBlur(img, (7, 7), 0)

        #Indices of detected people in outputs
        instances = []
        indices = []
        # indices = np.where(msg.has_image == 1)[0]
        # print("indices",indices)
        # masks = self.fix_masks_(instances.pred_masks.numpy(), indices)
        for i in range (len(msg.image_set)):
            x = msg.x[i]
            y = msg.y[i]
            img = msg.image_set[i]
            width = msg.width[i]
            height = msg.height[i]
            instance = (x,y,width,height)
            # print("instance",instance)
            # myRect = img[280:340, 330:390]
            instances.append(instance)
            # indices = msg.image_set[i]
        for y in range(len(msg.has_image)):
            ind = np.where(msg.has_image[y] == 1)[0]
            indices.append(ind)
            # if (msg.has_image[y]==1):
            #   indices = np.array(msg.has_image[y])
        #     else:
        #       continue
        print("2")
        print("LEN Indices", len(indices))
        # print("type instance",type(instances.astype(np.uint32)))
        # print("instances", instances)
        # print("indices", indices)
        #Parallel computation of color centers
        # centers = self.pool_.starmap(self.find_dom_color_, [(instances.pred_boxes.tensor[indices[ind_i]].numpy().astype(np.uint32), masks[ind_i], image_gb) for ind_i in range(len(indices))])
        centers = self.find_dom_color_(image_gb)
        # centers = pool.starmap(find_dom_color, [(instances.pred_boxes.tensor[indices[ind_i]].numpy().astype(np.uint32), masks[ind_i], image_gb, head_perc, upper_body_perc, 2) for ind_i in range(len(indices))])

        # print(type(centers))

        new_indices = []
        new_centers = []

        for ci, center in enumerate(centers):
            if not center is None:
                new_indices.append(indices[ci])
                new_centers.append(center)

        indices = new_indices
        centers = new_centers

        self.boxes = instances
        # self.masks = [instances.pred_masks.numpy()[ind] for ind in indices]

        if not self.centers:
            print("3A")
            self.centers = centers
            self.ids = [i for i in range(len(self.centers))]
        else:
            print("3B")
            #Calculate distances between previous and current centers
            all_dists = self.find_all_dists_(centers)
            matches = []

            #Pick minimum distance as match till no more possible matches are available
            while all_dists:
                matches.append(all_dists[0][1])
                all_dists = self.update_all_dists_(all_dists, all_dists[0][1])

            #Update center
            self.centers = self.update_cur_center_(matches, centers)
            self.ids = [-1 for i in range(len(self.boxes))]

            for match in matches:
                self.ids[match[0]] = match[1]

            marker = len(matches)
            for i, id in enumerate(self.ids):
                if id == -1:
                    self.ids[i] = marker
                    marker += 1

        self.visualize_tracking_(image_gb, indices, instances)



def main(args):
  ic = Track()
  rospy.init_node('image_track', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)