import matplotlib.pyplot as plt
import random
import numpy as np

L = 10
P = 5

class Robot:
    def __init__(self,pos, goal) -> None:
        self.pos = pos
        self.path = []
        self.start = pos
        self.goal = goal
        self.color = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
    
    def update_start_goal(self, start, goal):
        # x, y = goal[0], goal[1]
        self.pos = start
        self.start = start
        self.goal = goal

class Compute_node:
    def __init__(self, x, y, num_DU = 1, cap_DU = 2) -> None:
        self.x = x
        self.y = y
        self.num_DU = num_DU 
        self.cap_DU = cap_DU
        self.robots = []
    def increase_DU(self):
        """
        Do something to increase number of DU
        """
        self.num_DU += 1
    def not_enough_cm_node(self):
        '''
        check if there's enough compute node capacity
        '''
        if self.cap_DU * self.num_DU < len(self.robots):
            return False
        return True
        #TODO random pertubata white tile


class Control_node:
    def __init__(self, x, y, num_compute) -> None:
        # plan path to close to compute node
        # compute dis in heuristic
        self.x = x
        self.y = y
        self.compute_nodes = self.gen_compute_node(num_compute)
        self.color = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))

    def gen_compute_node(self, num_compute):
        compute_nodes = []
        for i in range(num_compute):
            x = random.uniform(max(0, self.x-L/P ), min(L, self.x+L/P ))
            y = random.uniform(max(0, self.y-L/P ), min(L, self.y+L/P ))
            num_DU = random.randint(1, 2)
            compute_nodes.append(Compute_node(x, y, num_DU))
        return compute_nodes
    
    def update_com_DU(self):
        # robots associate to compute should first be decided
        for cm in self.compute_nodes:
            if cm.not_enough_cm_node():
                cm.increse_DU()

            


class WareHouse:
    def __init__(self, grid_data, num_agent=2, num_control = 2) -> None:
        self.grid_width = len(grid_data[0])
        self.grid_height = len(grid_data)
        self.grid_data = np.array(grid_data)
        self.control_nodes = self.gen_control_nodes(num_control)
        self.changed_grid = []
        # self.robots = []
        # self.robots.append(Robot((0,0), (9,9)))
        self.num_agent = num_agent
        self.n_agents = num_agent
        self.action_dir = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        self.n_actions = len(self.action_dir)
        self.observation_space = len(self.action_dir)*10*10
        # self.robot = Robot([0, 0], [9, 9])
        self.robot_list = []
        self.robot_list.append(Robot([0, 0], [0, 5]))
        self.robot_list.append(Robot([4, 5], [3, 9]))
        assert self.num_agent == len(self.robot_list) 
        self.legal_start_end = [[x, y] for x in range(self.grid_height) for y in range(self.grid_width) if self.grid_data[x][y]==0]

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
    def plot_gird(self):
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid_data[i][j] == 1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='black', alpha = 1))
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black', alpha = 1))

        # for control in self.control_nodes:
        #     ax.scatter(control.x, control.y, color = control.color, marker='*', zorder = 10)
        #     # print(control.color)
        #     # print(control.x, control.y)
        #     for compute in control.compute_nodes:
        #         ax.scatter(compute.x, compute.y, color = control.color, marker=',', zorder = 10)


        # Set the aspect ratio to 'equal' for square grid cells
        ax.set_aspect('equal')

        # Customize the ticks and gridlines
        ax.set_xticks(range(self.grid_width+1))
        ax.set_yticks(range(self.grid_height+1))
        # ax.xaxis.tick_top()
        ax.tick_params(length=0, width=0.5)
        # Add gridlines
        ax.grid(color='black', linewidth=0.5)

        self.fig, self.ax = fig, ax
        # return fig, ax
        del fig, ax
        # plt.close()
        return self.fig, self.ax
    
    def plot_path(self, ax, path, color):
        for x, y in path:
            # ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor='green'))
            ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=color))
        return ax
    
    def render(self):
        fig, ax = self.plot_gird()
        for robot in self.robot_list:
            pos = robot.pos
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor=robot.color, alpha = 1))

        return fig, ax
    
    def gen_control_nodes(self, num_control):
        control_nodes = []
        for i in range(num_control):
            x = random.uniform(0, L)
            y = random.uniform(0, L)
            num_compute = 3
            control_nodes.append(Control_node(x, y, num_compute))
        return control_nodes

    def man_dis(self, pos1, pos2):
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]) 

    def reset(self):
        for x, y in self.changed_grid:
            self.grid_data[x][y] = 0
        self.changed_grid = []
        self.legal_start_end = [[x, y] for x in range(self.grid_height) for y in range(self.grid_width) if self.grid_data[x][y]==0]
        
        start_end = random.sample(self.legal_start_end, self.n_agents*2)
        idx = 0
        for agent in self.robot_list:
            agent.update_start_goal(start_end[idx], start_end[idx+1])
            idx +=2
       
        obs = np.zeros((self.num_agent, 5, self.grid_height, self.grid_width))
        for i in range(self.num_agent):
            obstacle = self.grid_data
            agt_pos = np.zeros(self.grid_data.shape)
            for agt in self.robot_list:
                agt_pos[agt.pos[0]][agt.pos[1]] = 1
            neighbor_goal = np.zeros(self.grid_data.shape)
            for idx, agt in enumerate(self.robot_list):
                if idx != i:
                    neighbor_goal[agt.goal[0]][agt.goal[1]] = 1
            agt_goal = np.zeros(self.grid_data.shape)
            agt_goal[self.robot_list[i].goal[0]][self.robot_list[i].goal[1]] = 1
            agt_goal[self.robot_list[i].pos[0]][self.robot_list[i].pos[1]] = 1

            v_hat = np.zeros(self.grid_data.shape)
            v = np.array(self.robot_list[i].goal) - np.array(self.robot_list[i].pos)
            length = np.sqrt(v[0]**2+v[1]**2)
            if length != 0:    
                v = v / length
            # print(v)
            v_hat[0][0]  = v[0]
            v_hat[0][1]  = v[1]

            obs[i] = np.stack((obstacle, agt_pos, neighbor_goal, agt_goal, v_hat), axis=0)   
        return obs
    
    def get_reward(self, actions):
        reward_list = []
        is_legal_move_list = []

        for idx, robot in enumerate(self.robot_list):
            reward = 0
            is_legal_move = True
            action_set = self.action_dir
            new_pos = (robot.pos[0]+action_set[actions[idx]][0], robot.pos[1]+action_set[actions[idx]][1])
            if actions[idx] != 4 :
                reward -= 0.3
            else:
                if new_pos[0] != robot.goal[0] or new_pos[1] != robot.goal[1]:
                    reward -= 0.5
            if new_pos[0] < 0 or new_pos[0] > self.grid_height-1 or new_pos[1] < 0 or new_pos[1] > self.grid_width-1:
                reward -= 1
                is_legal_move = False
            else:
                if self.grid_data[new_pos[0]][new_pos[1]] == 1:
                    reward -= 1
                    is_legal_move = False
                
                if self.man_dis(new_pos, robot.goal) < self.man_dis(robot.pos, robot.goal):
                    reward += 0.3
                else:
                    reward -= 0.5

                # if new_pos[0] == robot.goal[0] and new_pos[1] == robot.goal[1]:
                #     reward = 100
                #     is_legal_move = True

            reward_list.append(reward)
            is_legal_move_list.append(is_legal_move) 
        # print(len(reward_list), len(is_legal_move_list))
        return reward_list, is_legal_move_list
    def move_robots(self, is_legal_move_list, act_list):
        # print(is_legal_move_list)
        for robot, is_legal_move, action in zip(self.robot_list, is_legal_move_list, act_list):
            if is_legal_move:
                act_set  = self.action_dir
                robot.pos[0] = robot.pos[0]+act_set[action][0]
                robot.pos[1] = robot.pos[1]+act_set[action][1]



    def are_finished(self):
        dones = []
        for robot in self.robot_list:
            dones.append(
                robot.pos[0] == robot.goal[0] and 
                robot.pos[1] == robot.goal[1] 
            )
        return dones
    
    def step(self, actions):
        reward, is_legal_move_list = self.get_reward(actions)

        # generate some change grid
        for _ in range(5):
            col = random.randint(0, self.grid_width-1)
            row = random.randint(0, self.grid_height-1)
            if self.grid_data[row][col] != 1:
                self.changed_grid.append((row, col))
        
        # update grid
        for row, col in self.changed_grid:
            self.grid_data[row][col] = 2 # different from 1 indicates temporary change

        # remove something in changed_grid
        for _ in range(random.randint(2,4)):
            if len(self.changed_grid) > 1:
                idx = random.randint(0, len(self.changed_grid)-1)
                self.changed_grid.pop(idx)

        
        self.move_robots(is_legal_move_list, actions)


        dones = self.are_finished()

        obs = np.zeros((self.num_agent, 5, self.grid_height, self.grid_width))
        # pos = np.zeros()
        for i in range(self.num_agent):
            obstacle = self.grid_data
            agt_pos = np.zeros(self.grid_data.shape)
            for agt in self.robot_list:
                agt_pos[agt.pos[0]][agt.pos[1]] = 1
            neighbor_goal = np.zeros(self.grid_data.shape)
            for idx, agt in enumerate(self.robot_list):
                if idx != i:
                    neighbor_goal[agt.goal[0]][agt.goal[1]] = 1
            agt_goal = np.zeros(self.grid_data.shape)
            agt_goal[self.robot_list[i].goal[0]][self.robot_list[i].goal[1]] = 1
            
            v_hat = np.zeros(self.grid_data.shape)
            v = np.array(self.robot_list[i].goal) - np.array(self.robot_list[i].pos)
            length = np.sqrt(v[0]**2+v[1]**2)
            if length != 0:    
                v = v / length
            # print(v)
            v_hat[0][0]  = v[0]
            v_hat[0][1]  = v[1]
            obs[i] = np.stack((obstacle, agt_pos, neighbor_goal, agt_goal, v_hat), axis=0)
            # print(obs[i].shape)
            assert obs[i].shape == (5, 10, 10)
        # print(obs)
        # return next_state, reward, done, info
        return obs, reward, dones, 'Hello'
        
     
        
        
# Define the grid data with passable and nonpassable grids
# grid_data = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ]

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Plot the grid
# for i in range(grid_height):
#     for j in range(grid_width):
#         if grid_data[i][j] == 1:
#             ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='black'))
#         else:
#             ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black'))

# # Set the aspect ratio to 'equal' for square grid cells
# ax.set_aspect('equal')

# # Customize the ticks and gridlines
# ax.set_xticks(range(grid_width+1))
# ax.set_yticks(range(grid_height+1))
# # ax.xaxis.tick_top()
# ax.tick_params(length=0, width=0.5)

# # Add gridlines
# ax.grid(color='black', linewidth=0.5)

# # Show the graph
# plt.show()

if __name__=='__main__':
    grid_data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    FIX_RANDOM_SEED = 1
    if FIX_RANDOM_SEED:
        random.seed(8764)
        np.random.seed(420)
    
    warehouse = WareHouse(grid_data, 2)
    warehouse.reset()
    for i in range (5):
        # print(i)
        actions = [0, np.random.randint(0, 4)]
        # print(actions)
        warehouse.step(actions)
        fig, ax = warehouse.render()
        plt.draw()
        plt.pause(0.5)

    warehouse.reset()
    for i in range (5):
        # print(i)
        actions = [0, np.random.randint(0, 4)]
        # print(actions)
        warehouse.step(actions)
        fig, ax = warehouse.render()
        plt.draw()
        plt.pause(0.5)
