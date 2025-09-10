import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time
import random
from gym.utils import seeding
import math
import cv2
import matplotlib.pyplot as plt
import os


class UR5eRopeEnv(gym.Env):
    def __init__(self, fps, step_episode, radius_rope, youngs_modulus,
                 client_id):
        super(UR5eRopeEnv, self).__init__()
        self.client_id = client_id
        self._bool_debug = False
        self.state_id = 0  # For saving the state.
        self.step_episode = step_episode
        self.weight_speed = 0.05  # weight for speed

        """CHECK HERE"""
        # Initialize PyBullet environment.
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        # Load plane at default position (0,0,0)
        self.PLANE = r"../plane/plane.urdf"  # "C:/Users/kawaw/python/pybullet/ur5-bullet/UR5/code/robot_RL_curriculum/plane/plane.urdf"
        self.plane = p.loadURDF(self.PLANE, physicsClientId=self.client_id)

        # set frictino
        p.changeDynamics(
            self.plane,
            linkIndex=-1,
            lateralFriction=0.5,  # default is 0.5
            spinningFriction=0.001,
            rollingFriction=0.001,
            physicsClientId=self.client_id,
        )

        # control Frequency
        self.frequency = fps  # 10 Hz or 500 Hz
        self.dt_sample = 1.0 / self.frequency  # sampling time. [s]
        p.setTimeStep(
            self.dt_sample, physicsClientId=self.client_id
        )  # Set the control frequency.


        # Define maximum steps per episode
        self.max_episode_steps = step_episode
        # Rope parameters
        self.rope_length = 0.5  # 0.4 m -> 0.5 m
        self.num_links = 50
        self.link_length = 0.01
        self.link_mass = 0.01  # kg/m -> 1.0

        # Young's modulus parameters for rope elasticity
        self.youngs_modulus = youngs_modulus  # Pa (Pascals) - default value
        self.radius_rope = radius_rope
        self.rope_cross_sectional_area = np.pi * (radius_rope)**2  # m²
        # Spring constant based on Young's modulus
        self.rope_spring_constant = (
            self.youngs_modulus * self.rope_cross_sectional_area /
            self.link_length
        )  # N/m
        self.rope_link_ids = []
        self.constraint_ids = []  # Store constraint IDs for updating
        self.counter_save_robot = 0
        self.prev_link_vel = (
            1e5  # previous link velocity. This is for estimating force and torque.
        )
        self.rope_mid_point = np.zeros(3)
        self.prev_link_vels = {}
        # camera setting
        # Camera parameters
        self.camera_distance = 1.2  # Distance from the target (3 meters)
        self.camera_pitch = 0.0  # Keep pitch at 0 degrees
        self.camera_yaw = 0.0  # Start yaw at 0 degrees
        self.camera_target = [0, 0, 0]  # Target point in world coordinates
        self.x_cam, self.y_cam, self.z_cam = 0, 0, 0
        self.p_cam_goal = np.zeros(3)

        # Action space: next target position
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )  # (x,y,z,rx,ry,rz) #adopt the rotation vector. q.x,q.y,q.z,q.w)

        # Observation space: [eef_pose(6),eef_vel(6), pos_human(3),vel_human(3),force_eef(3),torque_eef(3),rope_length(1), p_middle(3),vel_middle(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        self.counter_step = 0

        # Initialize state variables
        self.time_step = 1
        # self.reset()

    def seed(self, seed=None):
        """Sets the seed for the environment and randomness."""
        # if seed is None:
        #    seed = np.random.randint(0, 2**32 - 1)

        # np.random.seed(seed)  # Seed NumPy's random generator
        # random.seed(seed)     # Seed Python's random module

        # self.np_random, seed = seeding.np_random(seed)  # Optionally, for OpenAI Gym seeding compatibility

        return seed

    def load_rope(self, p0, pN):
        """
        Initialize the rope as a chain of links attached to the UR5e's end-effector.
        """

        self.rope_link_ids = []
        self.constraint_ids = []  # Clear constraint IDs

        link_visual_shape_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.03,
            length=self.link_length,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )  # Blue color

        link_collision_shape_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.03,
            height=self.link_length,
            physicsClientId=self.client_id,
        )

        prev_link_id = p.createMultiBody(
            baseMass=self.link_mass,
            baseCollisionShapeIndex=link_collision_shape_id,
            baseVisualShapeIndex=link_visual_shape_id,
            basePosition=p0,  # 3D robot's joint.
            physicsClientId=self.client_id,
        )

        # Attach the last rope link to the moving point
        self.root_link_id = prev_link_id
        self.create_moving_point_root(p0)

        self.rope_link_ids.append(prev_link_id)

        p_end = np.array(pN)
        p_st = np.array(p0)
        horizontal_vec = p_end - p_st
        horizontal_dist = np.linalg.norm(horizontal_vec)
        # Normalize horizontal direction
        dir_vec = horizontal_vec / horizontal_dist
        dp = (p_end - p_st) / self.num_links  # np.linalg.norm(p_end-p_st)
        sag_amplitude = self.rope_length * 0.2  # Tune this as needed

        # Create the remaining rope links
        for i in range(1, self.num_links):
            t = i / self.num_links

            link_pos = self.get_rope_point(
                p_st=p_st,
                dir_vec=dir_vec,
                horizontal_dist=horizontal_dist,
                sag_amplitude=sag_amplitude,
                t=t,
            )

            link_id = p.createMultiBody(
                baseMass=self.link_mass,
                baseCollisionShapeIndex=link_collision_shape_id,
                baseVisualShapeIndex=link_visual_shape_id,
                basePosition=link_pos,
                physicsClientId=self.client_id,
            )

            constraint_id = p.createConstraint(
                parentBodyUniqueId=prev_link_id,
                parentLinkIndex=-1,
                childBodyUniqueId=link_id,
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, -self.link_length / 2],
                childFramePosition=[0, 0, self.link_length / 2],
                physicsClientId=self.client_id,
            )

            # Store constraint ID for later updates
            self.constraint_ids.append(constraint_id)

            # Set constraint parameters for inextensible rope
            # ERP (Error Reduction Parameter) controls stiffness
            erp = 1.0  # Maximum stiffness for inextensible rope
            # Damping parameter to prevent oscillations
            damping = 0.2  # Increased damping for stability

            # For inextensible rope, use very high max force
            inextensible_max_force = 1e6  # Much higher than spring constant

            p.changeConstraint(
                constraint_id,
                [0, 0, 0],  # No change in position
                maxForce=inextensible_max_force,  # Very high force for inextensibility
                erp=erp,
                relativePositionTarget=0.0,  # Target distance between links
                physicsClientId=self.client_id
            )

            self.rope_link_ids.append(link_id)
            prev_link_id = link_id

        # Attach the last rope link to the moving point
        self.last_link_id = prev_link_id
        self.create_moving_point(pN)

    def get_rope_point(self, p_st, dir_vec, horizontal_dist, sag_amplitude, t):
        """Get the position of the rope point at time t."""
        return p_st + dir_vec * horizontal_dist * t + np.array([0, 0, sag_amplitude * np.sin(2 * np.pi * t)])

    def create_moving_point_root(self, p0):
        """Create a massless point that the first rope link is attached to.
        This point should follow the robot's end-effector position.
        """

        self.point_id_root = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            basePosition=p0,
            physicsClientId=self.client_id,
        )

        # Attach the first rope link to the massless point
        p.createConstraint(
            parentBodyUniqueId=self.root_link_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.point_id_root,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,  # Fixed joint to keep them together
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -self.link_length / 2],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client_id,
        )

        # Note: The massless point will be updated to follow the robot's end-effector
        # in the update_target_position_root() method

    def create_moving_point(self, pN):
        """Create a massless point that the last rope link is attached to."""
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius_rope,
            rgbaColor=[0.0, 1.0, 0.0, 1.0],  # lightblue
            physicsClientId=self.client_id,
        )

        self.point_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=sphere_visual,
            basePosition=pN,
            physicsClientId=self.client_id,
        )

        # Attach the last rope link to the moving point
        p.createConstraint(
            parentBodyUniqueId=self.last_link_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.point_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,  # Allow the link to move #p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -self.link_length / 2],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client_id,
        )

    def update_target_position_root(self, p0):
        """Update the massless point position to follow the robot's end-effector."""
        # Update the massless point's position to follow the robot's end-effector
        p.resetBasePositionAndOrientation(
            self.point_id_root,
            p0,
            [0, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        # p.resetBasePositionAndOrientation(self.root_link_id, self.pose_eef[:3], [0, 0, 0, 1],physicsClientId=self.client_id)

    def update_target_position(self, pN):
        """Update the target position based on current speed and time step."""

        # Update the moving point's position in the simulation
        p.resetBasePositionAndOrientation(
            self.point_id,
            pN,
            [0, 0, 0, 1],
            physicsClientId=self.client_id,
        )  # (bodyUniqueID, posObj, oriObj(=quaternion))

    def step(self, p0, pN):
        """Apply action, update simulation, calculate reward, and return observation."""

        #Update the initial position of the rope.
        self.update_target_position_root(p0)
        #Update the end point of the rope.
        self.update_target_position(pN)

        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)
        self.time_step += 1
        ### done.

        ##### Time: t+1 :: POSTPROCESS ###############################
        ## Get midpoint of the rope
        self.rope_state = self.get_rope_state()

        # Check termination conditions
        done = False
        if self.time_step >= self.max_episode_steps:
            done = True

        return self.rope_state, done

    def get_rope_state(self):
        """Get the state of the rope."""
        pos_rope_list = []
        for i in range(self.num_links):
            pos_rope_list.append(p.getBasePositionAndOrientation(self.rope_link_ids[i], physicsClientId=self.client_id)[0])
        return pos_rope_list


    def calculate_rotation_angle_between_rotvecs(
        self, rotation_vector1, rotation_vector2
    ):
        """Calculate rotation angle from two rotation vectors.

        Args:
            rotation_vector1 (numpy.ndarray): current rotation vector (3)
            rotation_vector2 (numpy.ndarray): target rotation vector (3)

        Returns:
            angle (float) : rotation angle
            rotation_transform (numpy.ndarray) : normalized transform rotation vector (3)
        """
        # Clip rotation vector between 0 and 2 pi.  Is this useful?
        # rotation_vector1 = self.clip_rotation_vec(rotation_vector1)
        # rotation_vector2 = self.clip_rotation_vec(rotation_vector2)

        # Ensure numpy arrays of shape (3,1) for cv2.Rodrigues
        rotation_vector1 = np.asarray(rotation_vector1, dtype=np.float64).reshape(3, 1)
        rotation_vector2 = np.asarray(rotation_vector2, dtype=np.float64).reshape(3, 1)
        # print(f"rotation_vector1={rotation_vector1},rotation_vector2={rotation_vector2}")

        try:
            # Convert rotation vectors to rotation matrices
            R1, _ = cv2.Rodrigues(rotation_vector1)
            R2, _ = cv2.Rodrigues(rotation_vector2)

            # Check if Rodrigues conversion was successful
            if R1 is None or R2 is None:
                return 0.0, np.array([0.0, 0.0, 0.0])

            # Invert R1 (transpose works for rotation matrices)
            R1_inv = R1.T

            # Compute relative rotation matrix
            R12 = R2 @ R1_inv

            # Convert back to rotation vector
            rotation_transform, _ = cv2.Rodrigues(R12)

            # Check if Rodrigues conversion was successful
            if rotation_transform is None:
                return 0.0, np.array([0.0, 0.0, 0.0])

            # convert (3,1)>(3)
            rotation_transform = np.array(rotation_transform).squeeze()
            # clip rotationo vector between 0 and 2pi.
            rotation_transform = self.clip_rotation_vec(rotation_transform)

            # Handle full rotation (2*pi)
            if abs(2 * np.pi - np.linalg.norm(rotation_transform)) < 1e-6:
                rotation_transform = np.zeros(3)

            angle = np.linalg.norm(
                rotation_transform
            )  # calculate the rotation angle in radian.

            # print(f"rotataion_transform={rotation_transform},shape={rotation_transform.shape}")
            # normalize transform rotation vector.
            if angle > 1e-8:  # Avoid division by zero
                # Ensure rotation_transform is a numpy array and has the right shape
                rotation_transform = np.asarray(rotation_transform).flatten()
                rotation_transform = np.array(
                    [
                        rotation_transform[0] / angle,
                        rotation_transform[1] / angle,
                        rotation_transform[2] / angle,
                    ]
                )
            else:
                rotation_transform = np.array([0.0, 0.0, 0.0])

            return angle, rotation_transform

        except (cv2.error, ValueError, np.linalg.LinAlgError):
            # If any error occurs in the rotation calculations, return default values
            return 0.0, np.array([0.0, 0.0, 0.0])

    def quaternion_to_rotvec(self, q):
        """
        Convert a unit quaternion to a rotation vector (axis-angle).
        q.x=rx*sin(theta/2),q.y=ry*sin(theta/2),q.z=rz*sin(theta/2),qw=cos(theta/2)

        Parameters:
            q : array-like of shape (4,)
                The quaternion [x, y, z, w]

        Returns:
            rotvec : ndarray of shape (3,)
                The rotation vector
        """
        q = np.array(q, dtype=float)
        if q.shape != (4,):
            raise ValueError("Quaternion must be a 4-element vector [x, y, z, w]")

        x, y, z, w = q
        norm = np.linalg.norm(q)
        if not np.isclose(norm, 1.0):
            q /= norm
            x, y, z, w = q

        angle = 2 * np.arccos(w)
        sin_half_angle = np.sqrt(1 - w * w)

        if sin_half_angle < 1e-8:
            # When angle is close to 0, the axis is arbitrary
            return np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / sin_half_angle
            return axis  # angle * axis

    def quaternion_to_forward_vector(self, quarternion):
        """Convert a quaternion to a forward vector (assuming forward is +Z)."""
        [qx, qy, qz, qw] = quarternion
        x = 2 * (qx * qz + qw * qy)
        y = 2 * (qy * qz - qw * qx)
        z = 1 - 2 * (qx * qx + qy * qy)
        return np.array([x, y, z])

    def vector_angle(self, v1, v2):
        """Calculate the cosine value from the two rotatio vectors."""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-15)  # normalization
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-15)  # normalization
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return dot_product  # np.arccos(dot_product)

    def evaluate_alignment(self, quaternion, direction_vector):
        """Evaluate the alignment between a quaternion and a direction vector.
        Parameters:
        -------------
        quaternion : np.array
            rotation vector (3)
        direction_vector : np.array
            ideal orientation (3)
        """
        forward_vector = (
            quaternion.copy()
        )  # self.quaternion_to_forward_vector(quarternion=quaternion)

        angle = self.vector_angle(forward_vector, direction_vector)
        return angle  # [rad]

    def quaternion_rotate_vector(self, quarternion, vx, vy, vz):
        qx = quarternion[0]
        qy = quarternion[1]
        qz = quarternion[2]
        qw = quarternion[3]

        t2 = qw * qx
        t3 = qw * qy
        t4 = qw * qz
        t5 = -qx * qx
        t6 = qx * qy
        t7 = qx * qz
        t8 = -qy * qy
        t9 = qy * qz
        t10 = -qz * qz

        rx = vx * (2 * (t8 + t10) + 1) + vy * 2 * (t6 - t4) + vz * 2 * (t7 + t3)
        ry = vx * 2 * (t6 + t4) + vy * (2 * (t5 + t10) + 1) + vz * 2 * (t9 - t2)
        rz = vx * 2 * (t7 - t3) + vy * 2 * (t9 + t2) + vz * (2 * (t5 + t8) + 1)

        return rx, ry, rz

    def quaternion_multiply(self, qx1, qy1, qz1, qw1, quarternion):
        qx2 = quarternion[0]
        qy2 = quarternion[1]
        qz2 = quarternion[2]
        qw2 = quarternion[3]

        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2

        return [qx, qy, qz, qw]


    def rotation_vector_to_quaternion(self, rot_vec):
        """
        Convert rotation vector to quaternion.

        rot_vec: (3,) array-like, rotation vector (axis * angle)

        Returns:
            quat: (4,) numpy array, quaternion [x, y, z, w]
        """
        theta = np.linalg.norm(rot_vec)

        if theta < 1e-8:  # very small angle, avoid division by zero
            return np.array([0.0, 0.0, 0.0, 1.0])

        axis = rot_vec / theta
        half_theta = theta / 2.0
        sin_half_theta = np.sin(half_theta)

        qx = axis[0] * sin_half_theta
        qy = axis[1] * sin_half_theta
        qz = axis[2] * sin_half_theta
        qw = np.cos(half_theta)

        return np.array([qx, qy, qz, qw])

    def reset(self, rope_length, p0,pN,youngs_modulus,radius_rope):
        """Reset the environment for a new episode.
        bool_base: if False (for training), reset self.rosbag with the baseline model.
        if True (baseline), keep the self.rosbag to replayr the same situation.
        """
        self.time_step = 0

        # reset the simulation.
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(
            self.dt_sample, physicsClientId=self.client_id
        )  # Set the control frequency.
        # plane
        self.plane = p.loadURDF(self.PLANE, physicsClientId=self.client_id)

        p.changeDynamics(
            self.plane,
            linkIndex=-1,  # base index
            #lateralFriction=lateralFriction,  # default is 0.5
            #spinningFriction=spinningFriction,
            #rollingFriction=rolliingFriction,
            physicsClientId=self.client_id,
        )
        ########################################################

        # Initialize the rope properties. ###############
        # 1. Rope mass
        self.link_mass = 3.6e-4 # kg/0.01m
        # 2. Rope length
        self.rope_length = rope_length
        self.link_length = 0.01  # each link's length is 1 cm.
        self.num_links = int(self.rope_length / self.link_length)
        #################################################

        # load rope
        self.load_rope(p0, pN)

        #rope parameter setting.
        self.set_youngs_modulus(youngs_modulus)

        # camera position
        # robot's end-effector.
        p_eef = p0.copy()  # Get the position (x, y, z)
        # human's turning center
        p_h = np.array(pN)  # (x,y,z)
        # vector from the robot to human
        vec_eef2h = p_h - p_eef
        dist_eef2h = np.linalg.norm(vec_eef2h)
        u_eef2h = vec_eef2h / dist_eef2h  # unit vector.
        self.p_cam_goal = (p_h + p_eef) / 2.0
        # rotate 30 degree.
        theta = np.pi / 5
        scale = 3.0
        self.x_cam = (
            p_eef[0]
            + (u_eef2h[0] * np.cos(theta) - u_eef2h[1] * np.sin(theta)) * scale
        )
        self.y_cam = (
            p_eef[1]
            + (u_eef2h[1] * np.cos(theta) + u_eef2h[0] * np.sin(theta)) * scale
        )
        if self.x_cam==0.0 and self.y_cam==0.0:
            self.x_cam = 0.4
            self.y_cam = 0.4
        self.z_cam = max(p_eef[2],p_h[2]) + 0.3
        #################################################

        self.rope_state = self.get_rope_state()

        return self.rope_state

    def render(self, mode="human"):
        if mode == "rgb_array":
            # Move the camera 90 degrees in yaw
            self.camera_yaw = 90.0  # Update yaw to 90 degrees

            # Calculate new camera position
            camera_pos = [
                self.x_cam,
                self.y_cam,
                self.z_cam,
            ]  # self.get_camera_position()

            # Use pybullet's camera to capture an RGB image from the environment
            width, height, rgb_img, _, _ = p.getCameraImage(
                640,
                480,
                viewMatrix=p.computeViewMatrix(
                    cameraEyePosition=camera_pos,
                    cameraTargetPosition=self.p_cam_goal,
                    cameraUpVector=[0, 0, 1],  # Assuming Z is up
                    physicsClientId=self.client_id,
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=90,  # Field of view
                    aspect=640 / 480,
                    nearVal=0.1,
                    farVal=20,
                    physicsClientId=self.client_id,
                ),
                physicsClientId=self.client_id,
            )
            #print(f"camera_pos={camera_pos},self.p_cam_goal={self.p_cam_goal},width={width},height={height}")
            # Convert image to a NumPy array and reshape it to (height, width, 4) (RGBA format)
            rgb_img = np.array(rgb_img, dtype=np.uint8)
            rgb_img = rgb_img.reshape((height, width, 4))

            # Remove the alpha channel (RGBA -> RGB)
            rgb_img = rgb_img[:, :, :3]
            # print("rgb_img.shape=",rgb_img.shape)
            if not isinstance(rgb_img, np.ndarray):
                rgb_img = np.array(rgb_img)

            #plt.imshow(rgb_img)
            return rgb_img
        else:
            pass  # self.env.render(mode)

    def set_youngs_modulus(self, youngs_modulus):
        """
        Set the Young's modulus of the rope and update all constraint parameters.

        Args:
            youngs_modulus (float): Young's modulus in Pa (Pascals)
        """
        self.youngs_modulus = youngs_modulus
        # Recalculate spring constant based on new Young's modulus
        self.rope_spring_constant = (
            self.youngs_modulus * self.rope_cross_sectional_area /
            self.link_length
        )

        # Update all rope constraints with new parameters
        if hasattr(self, 'constraint_ids') and len(self.constraint_ids) > 0:
            # Update all stored constraint IDs with new spring constant
            for constraint_id in self.constraint_ids:
                p.changeConstraint(
                    constraint_id,
                    [0, 0, 0],  # No change in position
                    maxForce=self.rope_spring_constant,
                    erp=0.8,  # Stiffness parameter
                    relativePositionTarget=0.0,
                    physicsClientId=self.client_id
                )

    def get_youngs_modulus(self):
        """
        Get the current Young's modulus of the rope.

        Returns:
            float: Current Young's modulus in Pa
        """
        return self.youngs_modulus

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(physicsClientId=self.client_id)

    # Function to calculate camera position based on distance, pitch, and yaw
    def get_camera_position(self):
        pitch_rad = math.radians(self.camera_pitch)
        yaw_rad = math.radians(self.camera_yaw)

        x = (
            -self.camera_distance
        )  # self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.camera_distance  # * math.cos(pitch_rad) * math.cos(yaw_rad)
        z = 1.0  # self.camera_distance * math.sin(pitch_rad)

        return [x, y, z]
