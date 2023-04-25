import json
from collections import defaultdict
import random

class Bot(object):
    """
    The Bot class that applies the Qlearning logic to Flappy bird game
    After every iteration (iteration = 1 game that ends with the bird dying) updates Q values
    After every DUMPING_N iterations, dumps the Q values to the local JSON file
    """

    def __init__(self):
        self.gameCNT = 0
        self.DUMPING_N = 25
        self.discount = 1.0 #[This is for students to fill in]
        self.r = {0: 0, 1: -1000}  # Reward function for students to fill in: Solution for question 2
        self.lr = 0.7 # learning rate for students to fill in
        self.load_qvalues()
        self.last_state = "420_240_0"
        self.last_action = 0
        self.moves = []
        self.sigma = 0.01



    def load_qvalues(self):
        """
        Load q values from a JSON file
        """
        def init_values():
            return [0, 0]

        self.qvalues = defaultdict(init_values)
        try:
            fil = open("data/qvalues.json", "r")
        except IOError:
            return
        self.qvalues = json.load(fil)
        self.qvalues = defaultdict(init_values,**self.qvalues)
        fil.close()

    def act(self, states_arr):
        state = self.map_state(states_arr)

        # The below is for students to fill in
        # Solution for question 3

        self.moves.append(
            (self.last_state, self.last_action, state)
        )

        self.last_state = state

        prob = random.uniform(0,1)
        if prob < self.sigma:
            self.last_action = random.randint(0,1)
        else:
            self.last_action = 0 if self.qvalues[state][0] >= self.qvalues[state][1] else 1
        return self.last_action

    def update_scores(self, dump_qvalues=True):
        """
        This function is for students to fill in
        """
        history = list(reversed(self.moves))

        # Flag if the bird died in the top pipe
        high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False

        # Q-learning score updates
        t = 1
        for exp in history:
            state = exp[0]
            act = exp[1]
            res_state = exp[2]

            # Select reward
            if t == 1 or t == 2:
                cur_reward = self.r[1]
            elif high_death_flag and act:
                cur_reward = self.r[1]
                high_death_flag = False
            else:
                cur_reward = self.r[0]

            # Update
            self.qvalues[state][act] = (1-self.lr) * (self.qvalues[state][act]) + \
                                       self.lr * ( cur_reward + self.discount*max(self.qvalues[res_state]) )

            t += 1

        self.gameCNT += 1  # increase game count
        if dump_qvalues:
            self.dump_qvalues()  # Dump q values (if game count % DUMPING_N == 0)
        self.moves = []  # clear history after updating strategies

    def map_state(self, states_arr):
        # Question1 solution.
        playerx = states_arr[0]
        playery = states_arr[1]
        vel = states_arr[2]
        lowerPipes = states_arr[3]
        if -playerx + lowerPipes[0]["x"] > -50:
            curPipe = lowerPipes[0]
        else:
            curPipe = lowerPipes[1]


        xdif = -playerx + curPipe["x"]
        ydif = -playery + curPipe["y"]

        if xdif < 140:
            xdif = int(xdif) - (int(xdif) % 10)
        else:
            xdif = int(xdif) - (int(xdif) % 70)

        if -180 < ydif < 180:
            ydif = int(ydif) - (int(ydif) % 10)
        else:
            ydif = int(ydif) - (int(ydif) % 60)

        return str(int(xdif)) + "_" + str(int(ydif)) + "_" + str(vel)


    def dump_qvalues(self, force=False):
        """
        Dump the qvalues to the JSON file
        """
        if self.gameCNT % self.DUMPING_N == 0 or force:
            fil = open("data/qvalues.json", "w")
            json.dump(self.qvalues, fil)
            fil.close()
            print("Q-values updated on local file.")
