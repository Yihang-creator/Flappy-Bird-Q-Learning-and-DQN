# Flappy-Bird-Q-Learning-and-DQN

flappy.py: a UI visualization of the training process\
bot.py: the class implementing simple Q-Learning algorithm\
DQNAgent.py: the class implementing DQN algorithm\
Learn.py: Faster Training without UI.

To switch from q-learning to DQN, uncomment
# from DQNAgent import DQNAgent
# bot = DQNAgent()
comment out
from bot import Bot
bot = Bot()
