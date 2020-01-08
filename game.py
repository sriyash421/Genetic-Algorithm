import cv2
import numpy as np
from PIL import Image
import random
import time
import math

display = True
display_name = "game"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FPS = 400
W = 50
L = 50
pixels = 4


def draw(image, color, x, y):
    global L
    global W
    global pixels
    for i in range(pixels):
        for j in range(pixels):
            try:
                image[pixels*x+i][pixels*y+j] = color
            except:
                pass

    return image


def show(snake, food):
    global L
    global W
    global pixels
    image = np.zeros((L*pixels, W*pixels, 3), np.uint8)

    for i in range(0, L):
        image = draw(image, GREEN, i, 0)
        image = draw(image, GREEN, i, L-1)

    for i in range(0, W):
        image = draw(image, GREEN, 0, i)
        image = draw(image, GREEN, W-1, i)

    if not food.gen:
        x = food.pos[0][0]
        y = food.pos[0][1]
        image = draw(image, RED, x, y)

    # print("DRAWING")
    for i in range(snake.length):
        x = snake.body[i][0]
        y = snake.body[i][1]
        # print(snake.body)
        image = draw(image, WHITE, x, y)
    # print("=====")
    return image


class Snake():
    global L
    global W
    global pixels
    def __init__(self, WIDTH=W, LENGTH=L):

        self.start = np.array([1, 1])
        self.end = np.array([LENGTH-2, WIDTH-2])
        self.length = 3
        self.direction = "RIGHT"
        self.head = np.empty((1, 2), dtype=np.int32)
        self.head[0][:] = [random.randint(
            self.start[0]+2, self.end[0]), random.randint(self.start[1], self.end[1])]
        self.body = np.concatenate((self.head, self.head-[0, 1]), axis=0)
        self.body = np.concatenate((self.body, self.head-[0, 2]), axis=0)
        self.alive = True
        # print(self.body.shape)
        assert(self.length == self.body.shape[0] and self.body.shape[1] == 2)

    def move(self):
        # print("------")
        # for i in range(self.length):
        #     print(self.body[i])
        # print(self.direction)
        if self.direction == "UP":
            self.head += [-1, 0]
        elif self.direction == "RIGHT":
            self.head += [0, 1]
        elif self.direction == "DOWN":
            self.head += [1, 0]
        elif self.direction == "LEFT":
            self.head += [0, -1]
        self.body = np.concatenate(
            (self.head, self.body[0:self.length-1][:]), axis=0)
        # for i in range(self.length):
        #     print(self.body[i])
        # print("------")
        # assert(self.length == self.body.shape[0] and self.body.shape[1] == 2)

    def eat(self, food):

        new = np.empty((1, 2), dtype=np.int32)
        new[0][:] = 2*self.body[self.length-1] - self.body[self.length-2]
        self.body = np.concatenate((self.body, new), axis=0)
        self.length += 1
        assert(self.length == self.body.shape[0] and self.body.shape[1] == 2)

    def status(self):
        global L
        global W
        for i in range(self.length):
            part = self.body[i][:]
            if part[0] == 0 or part[0] == L or part[1] == 0 or part[1] == W:
                self.alive = False
                # print("wall")
                return
            for j in range(i+1, self.length):
                part2 = self.body[j][:]
                if np.array_equal(part, part2):
                    # print("bhida", i, j)
                    # print(part)
                    # print(part2)
                    self.alive = False
                    return

    # def get_dist(self, arr):
    #     head = np.copy(self.head[0][:])
    #     dis = 0
    #     global L
    #     global W
    #     while True:
    #         dis += 1
    #         head += arr
    #         if head[0] == 0 or head[0] == L or head[1] == 0 or head[1] == W:

    #             break
    #     return dis

    # def isfree(self, dir) :
    #     head = np.copy(self.head[0][:])
    #     count = 0
    #     while True :
    #         head += dir
    #         if head > W

    def get_obs2(self, food):
        global L
        global W
        temp = []
        dir = 0
        if self.direction == "UP" :
            dir = 0
        elif self.direction == "RIGHT" :
            dir = 1
        elif self.direction == "DOWN" :
            dir = 2
        if self.direction == "LEFT" :
            dir = 3
        # print(self.head.shape)
        RIGHT = self.head[0][0]+[[0, 1]]
        LEFT = self.head[0][0]+[[0, -1]]
        TOP = self.head[0][0]+[[-1, 0]]
        temp.append(float(dir))
        temp.append((LEFT[0][0] == 0 or LEFT[0][0] ==
                     L or LEFT[0][1] == 0 or LEFT[0][1] == W))
        temp.append(float(TOP[0][0] == 0 or TOP[0][0] ==
                          L or TOP[0][1] == 0 or TOP[0][1] == W))
        # print(RIGHT[0][0] == 0 or RIGHT[0][0] == L or RIGHT[0][1] == 0 or RIGHT[0][1] == W)
        temp.append(float(RIGHT[0][0] == 0 or RIGHT[0][0]
                          == L or RIGHT[0][1] == 0 or RIGHT[0][1] == W))
        x = self.head[0][0]-food.pos[0][0]
        y = self.head[0][1]-food.pos[0][1]
        temp.append((math.atan2(y, x)+math.pi/2)/(math.pi))
#        temp.append(float(self.length))
        # temp.append(0 if x > 0 else 0)


        return np.asarray(temp)

    # def get_obs(self, food):
    #     temp = []

    #     temp.append(self.get_dist(np.array([-1, 0])))
    #     temp.append(self.get_dist(np.array([-1, 1])))
    #     temp.append(self.get_dist(np.array([0, 1])))
    #     temp.append(self.get_dist(np.array([1, 1])))
    #     temp.append(self.get_dist(np.array([1, 0])))
    #     temp.append(self.get_dist(np.array([1, -1])))
    #     temp.append(self.get_dist(np.array([0, -1])))
    #     temp.append(self.get_dist(np.array([-1, -1])))

    #     dis = np.linalg.norm(self.head-food.pos)
    #     # print(self.head)
    #     # print(food.pos)
    #     x = self.head[0][0]-food.pos[0][0]
    #     y = self.head[0][1]-food.pos[0][1]

    #     temp.append(dis if x < 0 and y == 0 else 0)
    #     temp.append(dis if x < 0 and y > 0 else 0)
    #     temp.append(dis if x == 0 and y > 0 else 0)
    #     temp.append(dis if x > 0 and y > 0 else 0)
    #     temp.append(dis if x > 0 and y == 0 else 0)
    #     temp.append(dis if x > 0 and y < 0 else 0)
    #     temp.append(dis if x == 0 and y < 0 else 0)
    #     temp.append(dis if x < 0 and y < 0 else 0)

    #     x = self.head[0][0]-self.body[self.length-1][0]
    #     y = self.head[0][1]-self.body[self.length-1][1]

    #     temp.append(dis if x < 0 and y == 0 else 0)
    #     temp.append(dis if x < 0 and y > 0 else 0)
    #     temp.append(dis if x == 0 and y > 0 else 0)
    #     temp.append(dis if x > 0 and y > 0 else 0)
    #     temp.append(dis if x > 0 and y == 0 else 0)
    #     temp.append(dis if x > 0 and y < 0 else 0)
    #     temp.append(dis if x == 0 and y < 0 else 0)
    #     temp.append(dis if x < 0 and y < 0 else 0)

    #     return np.asarray(temp)


class Food():
    def __init__(self):
        self.pos = np.empty((1, 2), dtype=np.int32)
        self.gen = True

    def generate(self, x, y):
        self.pos[0][:] = np.array([random.randint(x[0], y[0]),
                                   random.randint(x[1], y[1])])


def game(policy, episode_length):
    global W
    global L
    start = time.time()
    fitness = 0.0
    snake = Snake()
    food = Food()
    food_dist = 999999
    if display:
        # cv2.namedWindow(display_name)
        image = show(snake, food)
        cv2.imshow(display_name, image)
    iter = 0
    while True:
        # print(iter)
        # print(snake.length)
        iter += 1
        now = time.time() - start
        # if now > episode_length*0.5 :
        #     print(now)
        if not snake.alive:
            fitness -= 1
            # print("Score : {}".format(fitness))
            try:
                cv2.destroyWindow(display_name)
            except:
                pass
            return fitness

        if now > episode_length:
            # print("Score : {}".format(fitness))
            try:
                cv2.destroyWindow(display_name)
            except:
                pass
            fitness = fitness
            return fitness

        if food.gen:
            food.gen = False
            food.generate((1, 1), (L-2, W-2))

        # observation_space = snake.get_obs(food)
        # print(observation_space)

        observation_space = snake.get_obs2(food)
        assert(observation_space.shape[0] == policy[0].shape[1])
        action_space = np.dot(policy[0], observation_space)
        for i in range(1, len(policy)):
            action_space = np.dot(policy[i], action_space)
        assert(action_space.shape[0] == 4)
        max_value = np.max(action_space)
        max_position = np.where(action_space == max_value)[0][0]
        
        # position = [2, 3, 0, 1]
        # max_position1 = input()#position[int(iter) % 4]
        # max_position = 0
        # if max_position1 == "W" :
        #     max_position = 0
        # if max_position1 == "A" :
        #     max_position = 3
        # if max_position1 == "S" :
        #     max_position = 2
        # if max_position1 == "D" :
        #     max_position = 1

        if max_position == 0:
            dir = "UP"
            # print("UP")
        elif max_position == 1:
            dir = "RIGHT"
            # print("RIGHT")
        elif max_position == 2:
            dir = "DOWN"
            # print("DOWN")
        elif max_position == 3:
            dir = "LEFT"
            # print("LEFT")

        # x = input()

        if snake.direction != "DOWN" and dir == "UP":
            snake.direction = "UP"
            # print("change UP")
        elif snake.direction != "UP" and dir == "DOWN":
            snake.direction = "DOWN"
            # print("change DOWN")
        elif snake.direction != "LEFT" and dir == "RIGHT":
            snake.direction = "RIGHT"
            # print("change RIGHT")
        elif snake.direction != "RIGHT" and dir == "LEFT":
            snake.direction = "LEFT"
            # print("change LEFT")
        # print(snake.head)
        # print(food.head)
        if np.array_equal(snake.head, food.pos) and not food.gen:
            food.gen = True
            snake.eat(food)
            fitness += 0.7
            # print("\n\n\n\n")
            # print("kaahahahbdbkjaldafvblasdbfkjhdsbflhdbfsadh")

        if not food.gen:
            temp_dist = np.linalg.norm(snake.head[0][:] - food.pos)

            if temp_dist < food_dist:
                fitness += 0.1
            else:
                fitness -= 0.2
            food_dist = temp_dist

        snake.move()
        snake.status()

        if display:
            image = show(snake, food)
            cv2.imshow(display_name, image)

        # time.sleep(10/1000)
        # if int(now) % 10 == 0 :
        cv2.waitKey(int(1000/FPS))

from result import policy
game(policy,1000)
