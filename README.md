# Deep Reinforcement Learning setup on Balloma Android game.

This work presents two Reinforcement Learning setups: a) Actor Critics Deep Deterministic Policy Gradients (DDPG) and b) Deep Q-Learning (DQN). DDPG's Actor and Critic components consists of a Convolutional and a Full Connected Neuronal Network respectively, the former infers agent's actions and the later assess quality of such predictions. For Deep Q-Learning it was constructed a ConvNet that represents the Q-Value function, mapping states to Q-Values used to select actions greedily with agent-environment further interactions without ever stopping exploration behavior (i.e GLIE). Similar setups were previously applied on 2D-world game playing agents, in this work it was applied to Balloma: a 3D-world android game. Deep Q-Learning performed better than DDPG, however currently further tunning is necessary in order to obtain a more practical video game playing agent.


*Please read `capstone_report.pdf` for the insigts on what has been developed in this repository.

## What's been used here:

   - Minicap.
   - Android Debug Bridge (adb)
   - Keras.
   - Tensorflow backend
   - Deep Determinist Policy Gradients
   - Deep Q-Learning
   - OpenCV

## I want to see something running:
   - Install python 3.7.5.
   - Activate developer mode on your android device.
   - Plug android device to PC and make sure it is usable by adb `adb devices` shows online device.
   - Install lib dependencies `pip install -r requirements.txt`.
   - Clone [minicap](https://github.com/openstf/minicap#usage)
   - Run minicap with `./run.sh autosize` (You need ndk-build for this)
   - Forward requests to minicap with `adb forward tcp:1313 localabstract:minicap`
   - Install [Balloma](https://play.google.com/store/apps/details?id=net.blackriverstudios.balloma&hl=en) game in your android device.
   - Open Balloma and start game's first scene.
   - Run `python training.py`.
   * Currently this project is only compatible with Samsung S8+ device.

   It will start inputting actions onto the device infered by the Actor's
   ConvNet. Scene is restarted automatically after every episode ends. Also you can use the scripts in `plots` folder of this repo to see training progress through metrics.
