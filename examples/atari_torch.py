# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import ale_py
import torch
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

from ncps.torch import CfC
from ncps.torch import LTC
from ncps.datasets.torch import AtariCloningDataset
from PIL import Image

import cv2
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean((-1, -2))  # Global average pooling
        return x


class ConvCfC(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = ConvBlock()
        # print('test')
        # import pdb;pdb.set_trace()
        self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)
        # self.rnn = LTC(256, 4, batch_first=True)
        # wiring = 

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Merge time and batch dimension into a single one (because the Conv layers require this)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)  # apply conv block to merged data
        # Separate time and batch dimension again
        x = x.view(batch_size, seq_len, *x.shape[1:])
        # import pdb; pdb.set_trace()
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        return x, hx


def eval(model, valloader):
    losses, accs = [], []
    model.eval()
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)
            # import pdb;pdb.set_trace()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(-1) == labels).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_one_epoch(model, criterion, optimizer, trainloader):
    running_loss = 0.0
    pbar = tqdm(total=len(trainloader))
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        labels = labels.view(-1, *labels.shape[2:])  # flatten
        outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
        pbar.update(1)
    pbar.close()


def run_closed_loop(model, env, num_episodes=None):
    obs = env.reset()[0]
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    index = 0
    with torch.no_grad():
        while True:
            # PyTorch require channel first images -> transpose data
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)

            # Observation seems to be already normalized, see: https://github.com/mlech26l/ncps/issues/48#issuecomment-1572328370
            # obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255.0
            # add batch and time dimension (with a single element in each)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            pred, hx = model(obs, hx)
            # remove time and batch dimension -> then argmax
            action = pred.squeeze(0).squeeze(0).argmax().item()
            if(action > 3):
                action = 3
            obs, r, done, _, _ = env.step(action)
            total_reward += r
            if done:
                video_writer.release()
                obs = env.reset()[0]
                hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    # Count down the number of episodes
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns

def run_closed_loop_test(model, env, video_writer, num_episodes=None):
    obs = env.reset()[0]
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    index = 0
    with torch.no_grad():
        while True:
            # PyTorch require channel first images -> transpose data
            obs_uint8 = (obs*255).astype(np.uint8)
            image = Image.fromarray(obs_uint8)
            image_rgb = image.convert('RGB')
            video_writer.write(cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR))
            
            if (index < 200):
                if(index%5==0):
                    image_rgb.save(f'./saved_image/{index}.png')
            if(index==4000):
                # import pdb;pdb.set_trace()
                print("release videowriter")
                video_writer.release()
            index += 1
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32)

            # Observation seems to be already normalized, see: https://github.com/mlech26l/ncps/issues/48#issuecomment-1572328370
            # obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255.0
            # add batch and time dimension (with a single element in each)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            pred, hx = model(obs, hx)
            # remove time and batch dimension -> then argmax
            action = pred.squeeze(0).squeeze(0).argmax().item()
            obs, r, done, _ = env.step(action)
            total_reward += r
            if done:
                # video_writer.release()
                obs = env.reset()[0]
                hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    # Count down the number of episodes
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns

if __name__ == "__main__":
    import os
    import re
    pattern = r".*\.png$"
    for file_name in os.listdir('./saved_image'):
        if re.match(pattern, file_name):
            os.remove(os.path.join('./saved_image', file_name))
    
    
    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)
    obs = env.reset()[0]
    width = obs.shape[0]
    height = obs.shape[1]
    video_writer = cv2.VideoWriter('breakout.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    
    train_ds = AtariCloningDataset("breakout", split="train")
    val_ds = AtariCloningDataset("breakout", split="val")
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, num_workers=16, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=32, num_workers=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:3")
    model = ConvCfC(n_actions=env.action_space.n)
    # print(env.action_space.n)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.000001)

    for epoch in range(30):  # loop over the dataset multiple times
        train_one_epoch(model, criterion, optimizer, trainloader)

        # Evaluate model on the validation set
        val_loss, val_acc = eval(model, valloader)
        print(f"Epoch {epoch+1}, val_loss={val_loss:0.4g}, val_acc={100*val_acc:0.2f}%")

        # Apply model in closed-loop environment
        returns = run_closed_loop(model, env, num_episodes=10)
        scheduler.step()
        print(f"Mean return {np.mean(returns)} (n={len(returns)})")
        torch.save(model.state_dict(), "breakout_model_cfc.pth")
    # Visualize Atari game and play endlessly
    check_point = torch.load('breakout_model_cfc.pth')
    model.load_state_dict(check_point)
    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)
    run_closed_loop_test(model, env, video_writer)
