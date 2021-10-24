import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

class Clean_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """


    def __init__(self, world_size,step_size,trash_size,obstacle_size,
                 trash_num,obstacle_num,max_step_num,obs_size):
        super(Clean_Env, self).__init__()

        # Size of the 1D-grid
        self.obs_size = obs_size
        self.world_size = world_size
        self.step_size = step_size
        self.trash_size = trash_size
        self.obstacle_size = obstacle_size
        self.trash_num = trash_num
        self.obstacle_num = obstacle_num
        self.max_step_num = max_step_num
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array([0,0])

        self.goal = np .array([world_size-1,world_size-1])

        self.count=0

        #self.observation=np.zeros((grid_size,grid_size))
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(obs_size, obs_size, 1), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize variable
        self.count = 0
        # Initialize variable，state is an array,size is grid_size*grid_size
        self.map_obstacle_position = np.zeros((self.obstacle_num,2))
        self.map_trash_position = np.zeros((self.trash_num,2))
        # Initialize map and location
        self.init_state()
        # get current location
        obs = self._get_obs()
        print(obs.shape)
        return obs

    def step(self, action):

        self.count = self.count + 1

        #print(action)

        reward = self.excute_action(action)

        #reward = self.get_reward(action)

        info = {}

        done = self.get_done()

        obs = self._get_obs()

        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def init_state(self):
        self.map_obstacle_position = np.random.randint(low=0,high=self.world_size,
                                           size=(self.obstacle_num,2))

        self.map_trash_position = np.random.randint(low=0,high=self.world_size,
                                           size=(self.trash_num,2))

        self.agent_pos = np.array([0,0])

        return

    def _get_obs(self):

        full_img = np.zeros([self.world_size, self.world_size])

        obs_full = np.zeros([self.world_size + self.obs_size,self.world_size + self.obs_size])


        full_img[int(self.agent_pos[0]), int(self.agent_pos[1])] = 50

        for i in range(self.trash_num):
            left_bound= int(self.map_trash_position[i, 0] - self.trash_size / 2) \
                if int(self.map_trash_position[i, 0] - self.trash_size / 2) > 0 else 0

            right_bound = int(self.map_trash_position[i,0] + self.trash_size/2) \
                if int(self.map_trash_position[i,0] + self.trash_size/2) <= (self.world_size-1) else (self.world_size-1)
            up_bound = int(self.map_trash_position[i, 1] - self.trash_size / 2) \
                if int(self.map_trash_position[i, 1] - self.trash_size / 2) > 0 else 0
            bottom_bound = int(self.map_trash_position[i,1] + self.trash_size/2) \
                if int(self.map_trash_position[i,1] + self.trash_size/2) <= (self.world_size-1) else (self.world_size-1)

            full_img[ left_bound:right_bound
                      ,
                     up_bound:bottom_bound] = 255
        for j in range(self.obstacle_num):

            left_bound = int(self.map_obstacle_position[j, 0] - self.obstacle_size / 2) \
                if int(self.map_obstacle_position[j, 0] - self.obstacle_size / 2) > 0 else 0

            right_bound = int(self.map_obstacle_position[j, 0] + self.obstacle_size / 2) \
                if int(self.map_obstacle_position[j, 0] + self.obstacle_size / 2) <= (self.world_size - 1) else (
                        self.world_size - 1)
            up_bound = int(self.map_obstacle_position[j, 1] - self.obstacle_size / 2) \
                if int(self.map_obstacle_position[j, 1] - self.obstacle_size / 2) > 0 else 0
            bottom_bound = int(self.map_obstacle_position[j, 1] + self.obstacle_size / 2) \
                if int(self.map_obstacle_position[j, 1] + self.obstacle_size / 2) <= (self.world_size - 1) else (
                        self.world_size - 1)

            full_img[left_bound:right_bound,
            up_bound:bottom_bound] = 100

        obs_full[int(self.obs_size/2):self.world_size+int(self.obs_size/2),
        int(self.obs_size/2):self.world_size+int(self.obs_size/2)] = full_img


        left_bound = int(self.agent_pos[0] + self.obs_size / 2  - self.obs_size / 2) \
            if int(self.agent_pos[0] + self.obs_size / 2- self.obs_size / 2) > 0 else 0

        right_bound = int(self.agent_pos[0] + self.obs_size / 2 + self.obs_size / 2) \
            if int(self.agent_pos[0] + self.obs_size / 2 + self.obs_size / 2) <= (self.world_size - 1) else (
                self.world_size - 1)
        up_bound = int(self.agent_pos[1] + self.obs_size / 2 - self.obs_size / 2) \
            if int(self.agent_pos[1] + self.obs_size / 2 - self.obs_size / 2) > 0 else 0
        bottom_bound = int(self.agent_pos[1] + self.obs_size / 2 + self.obs_size / 2) \
            if int(self.agent_pos[1] + self.obs_size / 2 + self.obs_size / 2) <= (self.world_size - 1) else (
                self.world_size - 1)

        obs = obs_full[left_bound:
                     right_bound,
            up_bound:
            bottom_bound]

        obs = obs.reshape((self.obs_size,self.obs_size,1))
        return obs

    def excute_action(self, action):

        collide_flag = False

        clean_flag = False

        reward = 0

        self.agent_buf = self.agent_pos.copy()

        self.agent_pos = self.agent_pos+self.step_size * np.array([np.cos(action*np.pi)[0], np.sin(action * np.pi)[0]])

        if self.agent_pos[0] > self.world_size or self.agent_pos[1] > self.world_size or self.agent_pos[0] < 0 or (
                self.agent_pos[1] < 0):
            self.agent_pos = np.clip(self.agent_pos, 0, self.world_size)

        for i in range(self.obstacle_num):
            #print(i)
            #print(self.observation[:,i])
            ve = self.agent_pos - self.map_obstacle_position[i,:]
            #print(np.sqrt(np.matmul(ve,ve)))
            if (np.sqrt(np.matmul(ve, ve))) < self.obstacle_size:
                self.agent_pos = self.agent_buf
                collide_flag = True
                break
        for j in range(self.trash_num):

            ve = self.agent_pos - self.map_trash_position[j, :]

            if (np.sqrt(np.matmul(ve, ve))) < self.trash_size:

                self.trash_num -= 1

                np.delete(self.map_trash_position, j, 0)

                clean_flag = True

                reward = 1
                Hydrogel
                break



        return reward

    def get_done(self):
        done = False

        if self.trash_num <= 0:
            done = True
        if self.count > self.max_step_num:
            done = True
        return done





if __name__ == '__main__':


    env = Clean_Env(world_size = 200,step_size = 2,trash_size = 15,obstacle_size = 10,
                 trash_num = 3,obstacle_num = 5,max_step_num = 30,obs_size = 40)

    check_env(env)
    obs = env.reset()
    episode_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(action)
        episode_reward += reward
        if done:
            print("Reward:", episode_reward)
            episode_reward = 0.0
            obs = env.reset()
            break