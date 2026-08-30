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

        with open('scores.csv', 'a') as file:
            file.write(f'{episode}, {env.score}, {agent.eps}\n')
        print("GAME OVER")
        agent.epsilon_decay_func()
        if episode % 10 == 0:
            agent.update_target_memory()
        if episode % 250 == 0:
            agent.save_checkpoint(f'models/model_ep{episode}.pt', episode)

def replay_model(agent, env, start_episode, num_of_games=10):
    state = env.reset()
    done = False
    total_reward = 0
    episode = start_episode
    scores = []
    for _ in range(num_of_games):
        episode += 1
        print(f"GAME {episode}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_memory(state, action, reward, next_state, done)
            total_reward += reward
            steps += 1
            print(f'Step: {steps} | Reward: {reward} | Done: {done} | Action: {action} | Epsilon: {agent.eps}')
            state = next_state
        scores.append(env.score)
    print(f'ALL SCORES: {scores} | MEAN: {sum(scores) / len(scores)}')


def list_models(path):
    available_models = []
    for file in os.listdir(path):
        if file.endswith(".pt"):
            available_models.append(file)
    return available_models

def get_newest_model(path):
    newest = None
    max = 0
    for file in os.listdir(path):
        if file.endswith(".pt"):
            r = len(file) - 4
            while file[r] != 'p':
                r -= 1
            temp = int(file[r+1:(len(file)-4)])
            if temp > max:
                max = temp
                newest = file
    return newest

if __name__ == '__main__':
    PATH = 'models/'
    URL = 'https://subwayonline.io/subway-surfers.embed'
    while True:
        title = "Choose action you want to do: "
        options = ['Learn a model from beginning',
                   'Continue learning of an existing model',
                   'Watch a trained model play',
                   'Delete models']
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
                newest_model = get_newest_model(PATH)
                if newest_model:
                    question = f"You chose: {options[idx]}. Do you want to proceed?"
                    answers = [f'Continue learning. Latest model: {newest_model}',
                               'No. Go back to main menu']
                    answer, idx = pick(answers, question, indicator='->')
                    match idx:
                        case 0:
                            print("Soon your browser will open a game. Please use browser in fullscreen mode! "
                                  "Also press a button in the middle so you can see main menu of game!")
                            webbrowser.open(URL)
                            msg = input("When you will be ready, press enter and go back to your browser. "
                                        "Agent will start soon after...")
                            agent = Agent()
                            env = Environment()
                            saved_ep = agent.load_checkpoint(os.path.join(PATH, newest_model))
                            traininng_loop(agent, env, start_episode=saved_ep)
                        case 1:
                            continue
                else:
                    question = "No models found"
                    answer = ["Go back to main menu"]
                    _, idx = pick(answer, question, indicator='->')
                    if idx == 0:
                        continue
            case 2:
                question = f"You chose: {options[idx]}. Now select model:"
                answers = list_models(PATH)
                answers.append(f'Go back to main menu')
                answer, idx = pick(answers, question, indicator='->')
                if idx == len(answers)-1:
                    continue
                else:
                    print("Soon your browser will open a game. Please use browser in fullscreen mode! "
                          "Also press a button in the middle so you can see main menu of game!")
                    webbrowser.open(URL)
                    msg = input("When you will be ready, press enter and go back to your browser. "
                                "Agent will start soon after...")
                    agent = Agent()
                    env = Environment()
                    saved_ep = agent.load_checkpoint(os.path.join(PATH, answer))
                    print(f"Loaded model: {saved_ep}")
                    agent.eps = 0.0
                    replay_model(agent, env, start_episode=saved_ep)
            case 3:
                question = f"You chose: {options[idx]}. Now select models to delete: (select by pressing space)"
                answers = list_models(PATH)
                answers.append(f'Go back to main menu')
                select = pick(answers, question, multiselect=True, min_selection_count=1, indicator='->')
                selected_names = [item[0] for item in select]
                if 'Go back to main menu' in selected_names:
                    continue
                for model_name in selected_names:
                    full_path = os.path.join(PATH, model_name)
                    if os.path.exists(full_path):
                        os.remove(full_path)





