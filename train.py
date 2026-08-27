from mpmath import monitor
from shapely.speedups import available
from torch.distributed.elastic import agent
from environment import Environment
from Agent import Agent
import time
import os
import webbrowser
import pyautogui
from pick import pick
import keyboard
import mss
import numpy as np
import cv2

def traininng_loop(agent, env, start_episode=0):
    episode = start_episode

    while True:
        episode += 1
        print(f"GAME {episode}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action=agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_memory(state, action, reward, next_state, done)
            agent.learn()
            total_reward += reward
            steps += 1
            print(f'Step: {steps} | Reward: {reward} | Done: {done} | Action: {action} | Epsilon: {agent.eps}')
            state = next_state
        print("GAME OVER")

        agent.epsilon_decay_func()
        if episode % 10 == 0:
            agent.update_target_memory()
        if episode % 50 == 0:
            os.mkdir(f'models/{episode}Episodes')
            agent.save_checkpoint(f'models/{episode}Episodes/model_ep{episode}.pt')

def list_models(path):
    available_models = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith(".pt"):
                available_models.append(file)
    return available_models

if __name__ == '__main__':
    PATH = 'models/'
    URL = 'https://subwayonline.io/subway-surfers.embed'
    while True:
        title = "Choose action you want to do: "
        #Only learning from beginning for now. I will work on rest of them after I'll be somewhat satisfied with learning
        options = ['Learn a model from beginning',
                   'Continue learning of an existing model',
                   'Watch a trained model play']
        option, idx = pick(options, title, indicator='->')
        match idx:
            case 0:
                question = f"You chose: {options[idx]}. Do you want to proceed?"
                answers = ['Yes',
                        'No. Go back to main menu']
                answer, idx = pick(answers, question, indicator='->')
                match idx:
                    case 0:
                        print("Soon your browser will open a game. Please use browser in fullscreen mode! "
                              "Also press a button in the middle so you can see main menu of game!")
                        webbrowser.open(URL)
                        msg = input("When you will be ready, press enter and go back to your browser. "
                                    "Agent will start soon after...")
                        time.sleep(2)
                        agent = Agent()
                        env = Environment()
                        traininng_loop(agent, env)
                    case 1:
                        continue
            case 1:
                question = f"You chose: {options[idx]}. Do you want to proceed?"
                answers = ['Yes',
                           'No. Go back to main menu']
                answer, idx = pick(answers, question, indicator='->')
                match idx:
                    case 0:
                        agent = Agent()
                        env = Environment()
                        traininng_loop(agent, env)
                    case 1:
                        continue
            case 2:
                question = f"You chose: {options[idx]}. Now select model:"
                answers = list_models(PATH)
                answers.append(f'Go back to main menu')
                answer, idx = pick(answers, question, indicator='->')
                if idx == len(answers)-1:
                    continue
                else:
                    pass





