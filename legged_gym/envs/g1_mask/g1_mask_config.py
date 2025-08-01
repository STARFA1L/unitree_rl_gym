from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RoughCfg_Masked( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

        # name: left_hip_pitch_joint, i: 0
        # name: left_hip_roll_joint, i: 1
        # name: left_hip_yaw_joint, i: 2
        # name: left_knee_joint, i: 3
        # name: left_ankle_pitch_joint, i: 4
        # name: left_ankle_roll_joint, i: 5
        # name: right_hip_pitch_joint, i: 6
        # name: right_hip_roll_joint, i: 7
        # name: right_hip_yaw_joint, i: 8
        # name: right_knee_joint, i: 9
        # name: right_ankle_pitch_joint, i: 10
        # name: right_ankle_roll_joint, i: 11
        # name: waist_yaw_joint, i: 12
        # name: waist_roll_joint, i: 13
        # name: waist_pitch_joint, i: 14
        # name: left_shoulder_pitch_joint, i: 15
        # name: left_shoulder_roll_joint, i: 16
        # name: left_shoulder_yaw_joint, i: 17
        # name: left_elbow_joint, i: 18
        # name: left_wrist_roll_joint, i: 19
        # name: left_wrist_pitch_joint, i: 20
        # name: left_wrist_yaw_joint, i: 21
        # name: right_shoulder_pitch_joint, i: 22
        # name: right_shoulder_roll_joint, i: 23
        # name: right_shoulder_yaw_joint, i: 24
        # name: right_elbow_joint, i: 25
        # name: right_wrist_roll_joint, i: 26
        # name: right_wrist_pitch_joint, i: 27
        # name: right_wrist_yaw_joint, i: 28    
    
    class masked_dof():
        name = [
        'waist_yaw_joint',
        'waist_roll_joint',
        'waist_pitch_joint',
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_roll_joint',
        'left_wrist_pitch_joint',
        'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
        'right_wrist_roll_joint',
        'right_wrist_pitch_joint',
        'right_wrist_yaw_joint'
        ]
        id = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        num= 17
        # default_angle = [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0]


    class unmask_dof():
        name = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        ]
        id = [0,1,2,3,4,5,6,7,8,9,10,11]
        num = 12
        # default_angle = [0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0]
       
    
    class env(LeggedRobotCfg.env):
        # num_observations = 47
        num_observations = 234
        # num_privileged_obs = 50
        num_privileged_obs = 237
        num_rl_actions = 12
        num_pd_actions = 4


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_mod.urdf'
        name = "g1_mask"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        # expected_dof=29
        # num_unmask_dof=12
        # num_masked_dof=17
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 1.0 #(my setting) #0.0(default setting)
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO_Masked( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_mask'

  
