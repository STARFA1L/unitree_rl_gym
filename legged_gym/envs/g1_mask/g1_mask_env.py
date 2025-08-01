from legged_gym.envs.base.legged_robot import LeggedRobot

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from legged_gym.envs.g1_mask.g1_mask_config import G1RoughCfg_Masked, G1RoughCfgPPO_Masked

from legged_gym.utils.debugger import Debugger

class G1Robot_Masked(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        ##modify:
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        ##
        return noise_vec

    
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        # print("INIT:feet_pos: ", self.feet_pos)
        
    def _init_buffers(self):

        # super()._init_buffers()

        # Below is the initialization of buffers specific to G1Robot_Masked, modified from LeggedRobot._init_buffers()
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        assert self.num_dofs==self.num_dof,f"Assertion failed: num_dofs ({self.num_dofs}) is not equal to num_dof ({self.num_dof})"
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # print("actor_root_state.shape:", actor_root_state.shape)
        # print("actor_root_state:", actor_root_state)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # print("dof_state_tensor.shape:", dof_state_tensor.shape)
        # print("dof_state_tensor:", dof_state_tensor)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # print("net_contact_forces.shape:", net_contact_forces.shape)
        # print("net_contact_forces:", net_contact_forces)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        ### set self.num_dofs to the number of unmasked DOFs
        # self.num_dofs = G1RoughCfg_Masked.unmask_dof.num
        # self.num_dof=self.num_dofs

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # Debugger.dprint(Debugger(), "g1_mask_env.py","G1Robot_Masked._init_buffers","self.root_states.shape:", self.root_states.shape,"self.root_states:", self.root_states)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         
        self.masked_dof_state= self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dofs, :] # shape: num_envs, num_dofs, 2
        # Debugger.dprint(Debugger(), "g1_mask_env.py","G1Robot_Masked._init_buffers","self.masked_dof_states.shape:", self.masked_dof_state.shape,"self.masked_dof_states:", self.masked_dof_state)
        self.dof_pos = self.masked_dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        # self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[...,:G1RoughCfg_Masked.unmask_dof.num, 0]
        # Debugger.dprint(Debugger(), "g1_mask_env.py","G1Robot_Masked._init_buffers","self.dof_pos.shape:", self.dof_pos.shape,"self.dof_pos:", self.dof_pos)
        self.dof_vel = self.masked_dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        # self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[...,:G1RoughCfg_Masked.unmask_dof.num, 1]
        # Debugger.dprint(Debugger(), "g1_mask_env.py","G1Robot_Masked._init_buffers","self.dof_vel.shape:", self.dof_vel.shape,"self.dof_vel:", self.dof_vel)
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        ##modify:
        if self.cfg.terrain.measure_heights:
                self.height_points = self._init_height_points()
        self.measured_heights = 0
        print("self.height_points.shape:",self.height_points.shape)
        ##

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # self.default_dof_pos = torch.zeros(G1RoughCfg_Masked.unmask_dof.num, dtype=torch.float, device=self.device, requires_grad=False)
        print("LeggedRobot._init_buffers: self.default_dof_pos.shape:", self.default_dof_pos.shape)
        # print("LeggedRobot._init_buffers: self.dof_names:", self.dof_names)
        # for i in range(self.num_dofs):
        #     name = self.dof_names[i]
        #     print(f"LeggedRobot._init_buffers: name: {name}, i: {i}")

        # for i in range(G1RoughCfg_Masked.unmask_dof.num):
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # # 如果name不在masked_dof.name中则跳过当前循环
            # if name in G1RoughCfg_Masked.masked_dof.name:
            #     continue
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        # print("UPDATE:feet_pos: ", self.feet_pos)
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        # print("pre obs cmp:self.obs_buf.shape=",self.obs_buf.shape)
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # print("in obs cmp:self.obs_buf.shape=",self.obs_buf.shape)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # print("heights.shape=",heights.shape)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            # print("heights added:self.obs_buf.shape=",self.obs_buf.shape)
            # print("heights added:self.privileged_obs_bu.shape=",self.privileged_obs_buf.shape)
        # add noise if needed
        if self.add_noise:
            # print("add noise:self.obs_buf.shape=",self.obs_buf.shape)
            # print("add noise:self.noise_scale_vec.shape=",self.noise_scale_vec.shape)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # print("after cmp:self.obs_buf.shape=",self.obs_buf.shape)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print("====================================\n","actions.shape:", self.actions.shape)
        # print("actions:", self.actions,"\n====================================\n")
        # de=Debugger()
        # de.dprint(de,"g1_mask_env.py", "step", "actions.shape:", self.actions.shape,"\nactions:", self.actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # de.dprint(de, "g1_mask_env.py", "step", "self.torques.shape:", self.torques.shape, "\nself.torques:", self.torques)
            self.full_torques = torch.cat([self.torques, torch.zeros(self.torques.size(0), G1RoughCfg_Masked.masked_dof.num, device=self.torques.device, dtype=self.torques.dtype)], dim=1)
            # de.dprint(de, "g1_mask_env.py", "step", "self.full_torques.shape:", self.full_torques.shape, "\nself.full_torques:", self.full_torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.full_torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _process_dof_props(self, props, env_id):
            """ Callback allowing to store/change/randomize the DOF properties of each environment.
                Called During environment creation.
                Base behavior: stores position, velocity and torques limits defined in the URDF

            Args:
                props (numpy.array): Properties of each DOF of the asset
                env_id (int): Environment id

            Returns:
                [numpy.array]: Modified DOF properties
            """
            ### set self.num_dofs to the number of unmasked DOFs
            self.num_dofs = G1RoughCfg_Masked.unmask_dof.num
            self.num_dof=self.num_dofs
            # Debugger.dprint(Debugger(), "g1_mask.py", "_process_dof_props", "env_id:", env_id,"len(props)",len(props))
            if env_id==0:
                self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
                self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                # Debugger.dprint(Debugger(), "g1_mask.py", "_process_dof_props", "self.torque_limits.shape:", self.torque_limits.shape, "self.torque_limits:", self.torque_limits)
                for i in range(min(len(props),self.num_dof)):
                    self.dof_pos_limits[i, 0] = props["lower"][i].item()
                    self.dof_pos_limits[i, 1] = props["upper"][i].item()
                    self.dof_vel_limits[i] = props["velocity"][i].item()
                    self.torque_limits[i] = props["effort"][i].item()
                    # soft limits
                    m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                    r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                    self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                    self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            return props

        
    def _compute_torques(self, actions):
            """ Compute torques from actions.
                Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
                [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

            Args:
                actions (torch.Tensor): Actions

            Returns:
                [torch.Tensor]: Torques sent to the simulation
            """
            #pd controller
            actions_scaled = actions * self.cfg.control.action_scale
            control_type = self.cfg.control.control_type
            if control_type=="P":
                print("Into G1Robot_Masked._compute_torques, control_type is P.\n")
                print("self.p_gains.shape:", self.p_gains.shape)
                print("self.d_gains.shape:", self.d_gains.shape)
                print("self.default_dof_pos.shape:", self.default_dof_pos.shape)
                print("self.dof_pos.shape:", self.dof_pos.shape)
                print("self.dof_vel.shape:", self.dof_vel.shape)
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            elif control_type=="V":
                torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            elif control_type=="T":
                torques = actions_scaled
            else:
                raise NameError(f"Unknown controller type: {control_type}")
            print("torque_limits.shape:", self.torque_limits.shape)
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.

        # Debugger.dprint(Debugger(), "g1_mask_env.py","G1Robot_Masked._reset_dofs","self.dof_pos.shape:", self.dof_pos.shape,"self.dof_state.shape:", self.dof_state.shape)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.12) * ~contact #my setting
        # pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact #default
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
        #my rew:
        # return 2.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    


