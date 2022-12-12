# library for implementing the user-interface for the soccer game
import numpy as np


NUM_PLAYERS = 2
NUM_DIMS = 2
FIELD_SIZE = (4, 2)
GATES = np.array([[[0, 0], [0, 1]], [[3, 0], [3, 1]]])
ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'STICK': 4
}
MAXINDEX = int('177', 8)
NACTIONS = 5


# define a game class
class SoccerGame:

    def __init__(self, start_coords, start_ball_pos):
        self.start_coords = start_coords.copy()
        self.start_ball_pos = start_ball_pos.copy()

    def restart(self):
        self.coords = np.zeros((NUM_PLAYERS, NUM_DIMS), dtype=np.int)
        self.ball_pos = np.zeros((NUM_PLAYERS, ), dtype=np.int)

        self.coords[0] = self.start_coords[0]
        self.coords[1] = self.start_coords[1]
        self.ball_pos[0] = self.start_ball_pos[0]
        self.ball_pos[1] = self.start_ball_pos[1]

    def get_coords(self):
        return self.coords

    def get_ball_pos(self):
        return self.ball_pos

    def reward(self):
        # From the perspective of A
        # if player A has the ball
        if self.ball_pos[0]:
            if np.equal(self.coords[0], GATES[0]).all(axis=1).any():
                return 100
            elif np.equal(self.coords[0], GATES[1]).all(axis=1).any():
                return -100
            else:
                return 0

        # if player B has the ball
        else:
            if np.equal(self.coords[1], GATES[0]).all(axis=1).any():
                return 100
            elif np.equal(self.coords[1], GATES[1]).all(axis=1).any():
                return -100
            else:
                return 0

    @staticmethod
    def act_to_next(coords, a):
        # this could work even if coords is numpy array
        x, y = coords
        # get code
        code = a
        if code == 0:
            y = min(FIELD_SIZE[1] - 1, y + 1)
        elif code == 1:
            y = max(0, y - 1)
        elif code == 2:
            x = max(0, x - 1)
        elif code == 3:
            x = min(FIELD_SIZE[0] - 1, x + 1)

        return np.array([x, y], dtype=np.int)

    def step(self, actA, actB):

        # let's move first
        coords_A = self.coords[0].copy()
        coords_B = self.coords[1].copy()
        want_to_A = self.act_to_next(coords_A, actA)
        want_to_B = self.act_to_next(coords_B, actB)

        # let's flip the coin
        flip = np.random.choice(2)
        # flip = 1
        # if A moves first
        if flip:
            # Does A moves into a collision?
            if np.all(want_to_A == coords_B):
                # Does A has the ball?
                if self.ball_pos[0] == 1:
                    self.ball_pos[0] = 0
                    self.ball_pos[1] = 1
            else:
                self.coords[0] = want_to_A.copy()

            # Does B moves into a collision?
            if np.all(want_to_B == self.coords[0]):
                # Does B has the ball?
                if self.ball_pos[1] == 1:
                    self.ball_pos[0] = 1
                    self.ball_pos[1] = 0
            else:
                self.coords[1] = want_to_B.copy()

        # if B moves first
        else:
            # Does B moves into a collision?
            if np.all(want_to_B == coords_A):
                # Does B has the ball?
                if self.ball_pos[1] == 1:
                    self.ball_pos[0] = 1
                    self.ball_pos[1] = 0
            else:
                self.coords[1] = want_to_B.copy()

            # Does A moves into a collision?
            if np.all(want_to_A == self.coords[1]):
                # Does A has the ball?
                if self.ball_pos[0] == 1:
                    self.ball_pos[0] = 0
                    self.ball_pos[1] = 1
            else:
                self.coords[0] = want_to_A.copy()

        rwd = self.reward()
        done = False
        if rwd != 0:
            done = True
        return self.coords, self.ball_pos, rwd, -rwd, done


def get_state_index(coords, ball_pos):
    ax, ay = coords[0]
    bx, by = coords[1]
    C = ball_pos[0]

    A = ay * FIELD_SIZE[0] + ax
    B = by * FIELD_SIZE[0] + bx

    ret = ''.join(map(str, [C, A, B]))
    return int(ret, 8)


def encode_action_combination(actA, actB, nact):
    return actA * nact + actB


def decode_action_combination(comb, nact):
    actA = comb // nact
    actB = comb % nact
    return actA, actB


if __name__ == '__main__':
    start_coords = np.array([[2, 1], [1, 1]], dtype=np.int)
    start_ball_pos = np.array([0, 1], dtype=np.int)
    game = SoccerGame(start_coords, start_ball_pos)
    game.restart()
    while game.reward() == 0:
        print('Player A: {}'.format(game.coords[0]))
        print('Player B: {}'.format(game.coords[1]))
        if game.ball_pos[0] == 1:
            print('A has the ball')
        else:
            print('B has the ball')
        print()
        command = input()
        actA, actB = command.split(sep=',')
        game.step(ACTIONS[actA], ACTIONS[actB])
    print('Player A: {}'.format(game.coords[0]))
    print('Player B: {}'.format(game.coords[1]))

    print(get_state_index(start_coords, start_ball_pos))
