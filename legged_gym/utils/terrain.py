import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

##debug
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        # with open("normal_tmp_log.txt", "w") as f:
        #     print(f"Terrain:self.proportions: {self.proportions}", file=f, flush=True)

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        #debug:
        # if cfg.dump:
        #     time_now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     cfg.terrain_config_path = Path("dump_"+time_now+".yaml")
            # with open(cfg.terrain_config_path, 'w') as f:
            #     yaml.safe_dump({"terrains": []}, f)

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            # raise NotImplementedError
            self.curiculum()#i/10 as default
        elif cfg.selected:
            # raise NotImplementedError
            self.selected_terrain()
        # elif cfg.rebuild:
        #     raise NotImplementedError
        #     self.rebuild_terrain()
        else: 
            # raise NotImplementedError   
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        # raise NotImplementedError
        # with open("normal_tmp_log.txt", "a") as f:
        #     print(f"Using Terrain.randomized_terrain()", file=f, flush=True)
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.randomized_terrain:spawning to ({i},{j}),choice: {choice}, difficulty: {difficulty}", file=f, flush=True)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            # with open("normal_tmp_log.txt", "w") as f:
            #     print(f"Terrain:self.proportions: {self.proportions}", file=f, flush=True)
            # try:
            #     self._update_terrain_config(i, j, choice, difficulty)
            # except Exception as e:
            #     print(f"记录地形配置失败: {e}")
            #     # 可在此处添加回退逻辑（如写入临时日志）
    def curiculum(self):
        # raise NotImplementedError
        # with open("normal_tmp_log.txt", "a") as f:
        #     print(f"Using Terrain.curiculum()", file=f, flush=True)
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows #i/10 as default
                # disable curiculum(curriculum):
                # difficulty = 0.001
                choice = j / self.cfg.num_cols + 0.001 #i/20+0.001 as default

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
                # with open("normal_tmp_log.txt", "a") as f:
                #     print(f"Terrain.curiculum:spawning to ({i},{j}),choice: {choice}, difficulty: {difficulty}", file=f, flush=True)
                 # --- 新增配置记录逻辑 ---
                # try:
                #     self._update_terrain_config(i, j, choice, difficulty)
                # except Exception as e:
                #     print(f"记录地形配置失败: {e}")
                #     # 可在此处添加回退逻辑（如写入临时日志）

    def selected_terrain(self):
        # raise NotImplementedError
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        if terrain_type not in ["gap_terrain", "pit_terrain","rough_sloped_terrain"]:
            terrain_type = f"terrain_utils.{terrain_type}"
        # with open("normal_tmp_log.txt", "a") as f:
        #     print(f"selected_terrain:terrain_type: {terrain_type}", file=f, flush=True)
        # print(f"selected_terrain:terrain_type: {terrain_type}, {self.cfg.terrain_kwargs}")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    # def _update_terrain_config(self, i: int, j: int, choice: float, difficulty: float):
    #     """
    #     更新 YAML 配置文件（原子操作）
    #     若坐标已存在则覆盖，否则新增条目
    #     """
    #     # 读取现有配置
    #     # print(f"_update_terrain_config:terrain_config_path: {self.cfg.terrain_config_path}")
    #     # print(f"_update_terrain_config:i={i}, j={j}, choice={choice}, difficulty={difficulty}")
    #     with open(self.cfg.terrain_config_path, 'r') as f:
    #         config = yaml.safe_load(f) or {"terrains": []}
        
    #     # 查找并更新/添加条目
    #     terrains: List[Dict] = config["terrains"]
    #     existing = next(
    #         (item for item in terrains if item["i"] == i and item["j"] == j),
    #         None
    #     )
    #     if existing:
    #         existing.update({
    #             "choice": float(choice),
    #             "difficulty": float(difficulty)
    #         })
    #     else:
    #         terrains.append({
    #             "i": int(i),
    #             "j": int(j),
    #             "choice": float(choice),
    #             "difficulty": float(difficulty)
    #         })

    #     # 写回文件
    #     with open(self.cfg.terrain_config_path, 'w') as f:
    #         yaml.safe_dump(config, f, sort_keys=False)

    # def _load_rebuild_config(self, config_path: str) -> Dict[Tuple[int, int], List[float]]:
    #     """加载并验证重建配置文件"""
    #     config_dict = {}
        
    #     try:
    #         with open(config_path, 'r') as f:
    #             config = yaml.safe_load(f)
                
    #         for idx, item in enumerate(config.get("terrains", [])):
    #             # 字段完整性检查
    #             required_keys = ["i", "j", "choice", "difficulty"]
    #             if missing := [k for k in required_keys if k not in item]:
    #                 raise ValueError(f"Config item {idx} missing fields: {missing}")
                
    #             # 坐标边界检查
    #             i, j = item["i"], item["j"]
    #             if not (0 <= i < self.num_rows and 0 <= j < self.num_cols):
    #                 raise IndexError(
    #                     f"坐标 ({i},{j}) 超出范围 "
    #                     f"(max: {self.num_rows-1}, {self.num_cols-1})"
    #                 )
                
    #             # 参数范围检查
    #             if not (0 <= item["choice"] <= 1 and 0 <= item["difficulty"] <= 1):
    #                 raise ValueError(
    #                     f"参数范围错误 @ ({i},{j}): "
    #                     f"choice={item['choice']}, difficulty={item['difficulty']} "
    #                     "(需在[0,1]范围内)"
    #                 )
                
    #             config_dict[(i, j)] = [item["choice"], item["difficulty"]]
                
    #     except FileNotFoundError:
    #         raise RuntimeError(f"重建配置文件不存在: {config_path}")
    #     except yaml.YAMLError:
    #         raise RuntimeError(f"配置文件格式错误: {config_path}")
    #         terrain_config
    #     return config_dict
    
    # def rebuild_terrain(self, rebuild_config_path: str):
    #     """
    #     根据新配置文件重建地形
        
    #     Args:
    #         rebuild_config_path: 地形重建配置文件路径
    #     """
    #     # 加载新配置文件
    #     new_terrain_dict = self._load_rebuild_config(rebuild_config_path)
        
    #     for i in range(self.num_rows):
    #         for j in range(self.num_cols):
    #             # 参数提取与验证
    #             params = new_terrain_dict.get((i, j), [0.0, 0.0])
    #             choice, difficulty = self._validate_params(i, j, params)
                
    #             # 地形生成
    #             terrain = self.make_terrain(choice, difficulty)
    #             self.add_terrain_to_map(terrain, i, j)
                
    #             # 记录日志
    #             # log_entry = (
    #             #     f"Coord ({i:02d},{j:02d}) | "
    #             #     f"Choice: {choice:.2f} | "
    #             #     f"Difficulty: {difficulty:.2f} | "
    #             #     f"Status: {'Custom' if (i,j) in new_terrain_dict else 'Default'}"
    #             # )
    #             # self._log_to_file(log_entry)
    
    def make_terrain(self, choice, difficulty):
        # raise NotImplementedError
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # slope = difficulty * 0.4
        slope = difficulty * 0.25
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn pyramid_sloped_terrain,slope: {slope}", file=f, flush=True)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn pyramid_sloped_terrain&random_uniform_terrain,slope: {slope}", file=f, flush=True)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn pyramid_stairs_terrain,step_height: {step_height}", file=f, flush=True)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn discrete_obstacles_terrain,discrete_obstacles_height: {discrete_obstacles_height}", file=f, flush=True)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn stepping_stones_terrain,stepping_stones_size: {stepping_stones_size},stone_distance: {stone_distance}", file=f, flush=True)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn gap_terrain,gap_size: {gap_size}", file=f, flush=True)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
            # with open("normal_tmp_log.txt", "a") as f:
            #     print(f"Terrain.make_terrain:choice:{choice}, spawn pit_terrain,pit_depth: {pit_depth}", file=f, flush=True)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        # Print the values of x1, x2, y1, and y2
        # print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

##modify:
def rough_sloped_terrain(terrain, slope, platform_size=3.):
    terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
    terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)


