import numpy as np
import sys
import random
import pygame
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
from keras.models import load_model
import tensorflow as tf
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()


PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

import os
def get_iterations(filename):
    no_extension = filename.split('.')[0]
    num_iterations = no_extension.split('model')[1]
    return num_iterations + '.txt'

def get_file():
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.h5'):
            return get_iterations(filename)

class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        self.prev = [0,0,0,0]

        # network for behavior cloning
        self.model2 = Sequential()
        self.model2.add(Dense(12, activation='relu', input_shape=(8,)))
        self.model2.add(Dense(9, activation='relu'))
        self.model2.add(Dense(7, activation='relu'))
        self.model2.add(Dense(5, activation='relu'))
        self.model2.add(Dense(3, activation='relu'))
        self.model2.add(Dense(2, activation='softmax'))
        adam = Adam(lr=0.001)
        self.model2.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
        self.model2 = load_model('flappy_bc.h5')
        self.data = []


    def collect_data(self, input_actions, iteration):
        """
        Collect flappy bird data. Extracts features (flappy bird location and distance to the two pipes on the screen)
        and saves data

        @param input_actions: the input_action of the trained dqn rl agent
        @param iteration: iteration agent is on (used to determine when to save to txt file)

        Saves data into "training.txt"
        """
        playerMidPos_x = self.playerx + IMAGES['player'][0].get_width() / 2
        playerMidPos_y = self.playery + IMAGES['player'][0].get_height() / 2
        if iteration%10000 == 0 and len(self.data) > 0:
            f=open('training.txt','ab')
            np.savetxt(f, np.array(self.data))
            f.close()
            self.data = []
        if self.upperPipes[0]['x'] - playerMidPos_x < 0: 
            pipeMidPos_y = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            pipeMidPos_y1 = self.upperPipes[1]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            distance_diff_x = self.upperPipes[0]['x'] - playerMidPos_x
            distance_diff_y =  pipeMidPos_y - playerMidPos_y
            distance_diff_x1 = self.upperPipes[1]['x'] - playerMidPos_x
            distance_diff_y1 =  pipeMidPos_y1 - playerMidPos_y
            self.data.append((self.prev[0], self.prev[1],self.prev[2],self.prev[3], distance_diff_x, distance_diff_y,distance_diff_x1, distance_diff_y1, input_actions[1]))
            self.prev = [distance_diff_x, distance_diff_y,distance_diff_x1, distance_diff_y1]
        else:
            pipeMidPos_y = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            distance_diff_x = self.upperPipes[0]['x'] - playerMidPos_x
            distance_diff_y =  pipeMidPos_y - playerMidPos_y
            self.data.append((self.prev[0], self.prev[1],self.prev[2],self.prev[3], distance_diff_x, distance_diff_y, 0, 0, input_actions[1]))
            self.prev = [distance_diff_x, distance_diff_y, 0, 0]

    def behavior_cloning_agent(self):
        """
        Extract features and run behavioral cloning agent

        Returns action of trained behavioral cloning agent
        """
        playerMidPos_x = self.playerx + IMAGES['player'][0].get_width() / 2
        playerMidPos_y = self.playery + IMAGES['player'][0].get_height() / 2
        if self.upperPipes[0]['x'] - playerMidPos_x < 0: 
            pipeMidPos_y = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            pipeMidPos_y1 = self.upperPipes[1]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            distance_diff_x = self.upperPipes[0]['x'] - playerMidPos_x
            distance_diff_y =  pipeMidPos_y - playerMidPos_y
            distance_diff_x1 = self.upperPipes[1]['x'] - playerMidPos_x
            distance_diff_y1 =  pipeMidPos_y1 - playerMidPos_y
            data = (self.prev[0], self.prev[1],self.prev[2],self.prev[3], distance_diff_x, distance_diff_y,distance_diff_x1, distance_diff_y1)
            self.prev = [distance_diff_x, distance_diff_y,distance_diff_x1, distance_diff_y1]
            key = np.argmax(self.model2.predict(np.array(data).reshape((1,8)),verbose = 0))
        
        else:
            pipeMidPos_y = self.upperPipes[0]['y'] + IMAGES['pipe'][0].get_height() + PIPEGAPSIZE/2
            distance_diff_x = self.upperPipes[0]['x'] - playerMidPos_x
            distance_diff_y =  pipeMidPos_y - playerMidPos_y
            data = (self.prev[0], self.prev[1],self.prev[2],self.prev[3], distance_diff_x, distance_diff_y, 0, 0)
            self.prev = [distance_diff_x, distance_diff_y, 0, 0]
            key = np.argmax(self.model2.predict(np.array(data).reshape((1,8)),verbose = 0))
        return key

    def frame_step(self, input_actions, rl_agent = 0, iteration = 0, data = 0):

        pygame.event.pump()
        save_data = 0
        reward = 0.1
        terminal = False
        if iteration == None:
            iteration = 0

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

       

        #Collect Data
        if data:
            self.collect_data(input_actions, iteration)
        
        #RL Agent 
        if int(rl_agent):
            if input_actions[1] == 1:
                if self.playery > -2 * PLAYER_HEIGHT:
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True
                    SOUNDS['wing'].play() 


        #Behavior Cloning Agent
        else:
            action = self.behavior_cloning_agent()
            if action:
                if self.playery > -2 * IMAGES['player'][0].get_height():
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True
                    SOUNDS['wing'].play()
            print action == input_actions[1]

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird


        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                #SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            SOUNDS['hit'].play()
            SOUNDS['die'].play()
            if data:
                f=open('training.txt','ab')
                np.savetxt(f, np.array(self.data))
                f.close()
                self.data = []
            terminal = True
            self.__init__()
            reward = -1
    

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
    
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        #print ("FPS" , FPSCLOCK.get_fps())
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    # random.seed(0)
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
