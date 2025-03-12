
from mapUtilities import *
from utilities import *
from numpy import cos, sin
import numpy as np


class particle:

    def __init__(self, pose, weight):
        self.pose = pose
        self.weight = weight

    def motion_model(self, v, w, dt):
        #TODO: Implement the motion model for the particle
        """
        v: linear velocity
        w: angular velocity
        dt: time step
        """
        theta = self.pose[2]

        self.pose[0] += (v/w)*(-sin(theta) + sin(theta + w * dt))
        self.pose[1] += (v/w)*(cos(theta) - cos(theta + w * dt))
        self.pose[2] += w * dt

    # TODO: You need to explain the following function to TA
    def calculateParticleWeight(self, scanOutput: LaserScan, mapManipulatorInstance: mapManipulator, laser_to_odom_transformation: np.array):

        "Comments are what I'm pretty sure these do"
        
        T = np.matmul(self.__poseToTranslationMatrix(), laser_to_odom_transformation) "Takes the current pose and converts into "

        _, scanCartesianHomo = convertScanToCartesian(scanOutput) "Converts the scanned area (lidar) to cartesian coordinates"
        scanInMap = np.dot(T, scanCartesianHomo.T).T 

        likelihoodField = mapManipulatorInstance.getLikelihoodField() "Takes likelihood field generated in mapUtilities"
        cellPositions = mapManipulatorInstance.position_2_cell(
            scanInMap[:, 0:2])

        lm_x, lm_y = likelihoodField.shape

        cellPositions = cellPositions[np.logical_and.reduce(
                (cellPositions[:, 0] > 0, -cellPositions[:, 1] > 0, cellPositions[:, 0] < lm_y,  -cellPositions[:, 1] < lm_x))] "Takes only positions in likelihood field limits"

        log_weights = np.log(
            likelihoodField[-cellPositions[:, 1], cellPositions[:, 0]])
        log_weight = np.sum(log_weights)
        weight = np.exp(log_weight)
        weight += 1e-10 
        
        "Takes the sum of the logs of the likelihood field to calculate the importance weight of current particle/position"

        self.setWeight(weight)

    def setWeight(self, weight):
        self.weight = weight

    def getWeight(self):
        return self.weight

    def setPose(self, pose):
        self.pose = pose

    def getPose(self):
        return self.pose[0], self.pose[1], self.pose[2]

    def __poseToTranslationMatrix(self):
        x, y, th = self.getPose()

        translation = np.array([[cos(th), -sin(th), x],
                                [sin(th), cos(th), y],
                                [0, 0, 1]])

        return translation
