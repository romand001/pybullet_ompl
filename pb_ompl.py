try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import pybullet as p
import utils
import time
import math
from simple_pid import PID
from itertools import product
import copy

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id) -> None:
        # Public attributes
        self.id = id

        # prune fixed joints
        all_joint_num = p.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        print(self.joint_idx)
        self.joint_bounds = []

        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)

class Turtlebot(PbOMPLRobot):

    def __init__(self, id) -> None:
        # Public attributes
        self.id = id

        # prune fixed joints
        all_joint_num = p.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        print(self.joint_idx)
        self.joint_bounds = []

        self.pos = [0, 0]

        # initialize PID to bring yaw error down to zero
        self.pid_yaw = PID(20.0, 0.0, 0.0, setpoint=0)

        self.reset()

    def get_cur_pos(self):
        return self.pos

    def smallest_signed_angle(self, x, y):
        a = (x - y) % (2 * math.pi)
        b = (y - x) % (2 * math.pi)
        return -a if a < b else b
    
    def track_goal(self, goal_xy):
        goal_x, goal_y = goal_xy

        cur_pos, cur_quat = p.getBasePositionAndOrientation(self.id)
        cur_orient = p.getEulerFromQuaternion(cur_quat)
        x = cur_pos[0]
        y = cur_pos[1]
        yaw = cur_orient[2]

        self.pos = [x, y]

        dx = goal_x - x
        dy = goal_y - y
        dist_err = math.sqrt(dx**2 + dy**2)
        # yaw_err = math.atan2(dy, dx) - yaw
        yaw_err = self.smallest_signed_angle(math.atan2(dy, dx), yaw)

        # mean wheel speed in rot/s
        min_speed = 0.0
        max_speed = 5.0
        move_angle = math.radians(10)
        mean_wheel_speed = max_speed - min(abs(yaw_err), move_angle) / move_angle * (max_speed - min_speed)
        mean_wheel_speed *= 2 * math.pi

        # compute difference in speed between right wheel and left wheel
        d_speed = self.pid_yaw(yaw_err)
        l_speed = mean_wheel_speed - d_speed
        r_speed = mean_wheel_speed + d_speed

        # print(
        #     f'cur_pos: ({x:.2f}, {y:.2f}, {yaw:.2f}),\tgoal_pos: ({goal_x:.2f}, {goal_y:.2f}),\tyaw_err: {yaw_err:.2f},\t'
        #     f'l_speed: {l_speed:.2f},\tr_speed: {r_speed:.2f}'
        # )

        # send speed commands
        p.setJointMotorControl2(
            self.id, 
            self.joint_idx[0],
            targetVelocity=l_speed,
            controlMode=p.VELOCITY_CONTROL,
            force=500
        )
        p.setJointMotorControl2(
            self.id, 
            self.joint_idx[1],
            targetVelocity=r_speed,
            controlMode=p.VELOCITY_CONTROL,
            force=500
        )


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, obstacles = []) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.robot_id = robot.id
        self.path_ids = []
        self.point_id = -1
        self.obstacles = obstacles
        print(self.obstacles)

        self.space = PbStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        for i in range(2):
            bounds.setLow(i, -5.5)
            bounds.setHigh(i, 5.5)
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT") # RRT by default

    def draw_path(self, path):

        add_z = lambda xy : [xy[0], xy[1], 0.02]

        for path_id in self.path_ids:
            p.removeUserDebugItem(path_id)
        
        self.path_ids = []

        for idx in range(len(path)-1):
            self.path_ids.append(
                p.addUserDebugLine(
                    lineFromXYZ=add_z(path[idx]), 
                    lineToXYZ=add_z(path[idx+1]), 
                    lineColorRGB=[1,0,0],
                    lineWidth=2,
                    lifeTime=0
                )
            )

    def draw_point(self, point):

        add_z = lambda xy : [xy[0], xy[1], 0.02]

        if self.point_id != -1:
            p.removeUserDebugItem(self.point_id)

        p.addUserDebugPoints(
            pointPositions=[add_z(point)],
            pointColorsRGB=[[0,1,0]],
            pointSize=5
        )

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):

        # bounding box of turtlebot
        aabb_min = [state[0]-0.18, state[1]-0.18, 0]
        aabb_max = [state[0]+0.18, state[1]+0.18, 0.42]

        overlap_ids = p.getOverlappingObjects(aabb_min, aabb_max)

        for overlap_id in overlap_ids:
            if overlap_id[0] in self.obstacles:
                return False

        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        # self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        all_links = frozenset(
            [item for item in utils.get_all_links(robot.id) if not item in allow_collision_links])
        moving_bodies = [(robot.id, all_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "DRRT":
            self.planner = og.DRRT(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
        plan a path to goal from the given robot start state
        '''
        print("start_planning")
        print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        if solved:
            # print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            # sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
        plan a path to gaol from current robot state
        '''
        cur_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        start = cur_pos[:2]
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)

    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i],force=5 * 240.)
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)

    def execute_step(self, goal_pos):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            goal_pos: x and y of goal state
        '''

        self.robot.track_goal(goal_pos)
        p.stepSimulation()
        time.sleep(0.01)



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]