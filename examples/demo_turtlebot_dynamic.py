import os.path as osp
import pybullet as p
import math
import sys
import time
import pybullet_data
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pb_ompl

class BoxDemo():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # load robot
        robot_id = p.loadURDF("models/turtlebot.urdf", (0,0,0), useFixedBase = 0)
        robot = pb_ompl.Turtlebot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("DRRT")

        # add obstacles
        self.add_initial_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_initial_obstacles(self):
        # classic gap problem
        self.add_box((-2.5, 5, 0), (2, 0.1, 0.2))
        self.add_box((2.5, 5, 0), (2, 0.1, 0.2))

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def pos_dist(self, pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def demo(self):

        start = (0, 0)
        goal = (0, 10)

        self.robot.set_state(start)

        path_to_goal = False
        blocked = False
        path_idx = -1
        while True:

            # block gap when robot gets close to it
            if not blocked and abs(self.pb_ompl_interface.robot.get_cur_pos()[1] - 5) < 1:
                self.add_box((0, 5, 0),(0.5, 0.1, 0.2))
                self.pb_ompl_interface.set_obstacles(self.obstacles)
                blocked = True
                path_to_goal = False

            # plan path to goal if we don't have one
            if not path_to_goal:
                res, path = self.pb_ompl_interface.plan(goal)
                path.reverse()
                path_to_goal = res
                if path_to_goal:
                    self.pb_ompl_interface.draw_path(path)
                    self.pb_ompl_interface.draw_point(path[0])
                path_idx = 0

            # step simulation if we have a path to the goal
            if path_to_goal:

                cur_pos = self.pb_ompl_interface.robot.get_cur_pos()

                if self.pos_dist(cur_pos, path[path_idx]) < 0.1:
                    path_idx += 1
                    if path_idx == len(path):
                        return res, path

                self.pb_ompl_interface.execute_step(path[path_idx])



if __name__ == '__main__':
    env = BoxDemo()
    env.demo()