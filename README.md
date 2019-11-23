# Udacity Capstone Project

DISCLARIMER: This project is under construction (i.e unstable).

The objective of this work is to construct a video game playing robot, through
the use of Deep Reinforcement Learning techniques. This repository presents
a PoC of a RL setup on Baloma Android Video Game.

## What's been used here:

    - Minicap.
    - Keras.
    - Deep Determinist Policy Gradients.
    - OpenCV

## What is currently working:

    - Environment permits inputting actions to game running in a Android device.
    - States are correctly captured (i.e game frames).
    - Determine if task is done (i.e episode ends) (not elegant solution implemented though).
    - Environment resetting (not elegant solution implemented though).
    - DDPG Agent.

## I want to see something running:
    - Activate developer mode in your android device.
    - Plug android device to PC and make sure it is usable by adb `adb devices` shows online device.
    - Install lib dependencies `pip install -r requirements.txt`.
    - Configure and start [minicap](https://github.com/openstf/minicap#usage) (that's actually easy).
    - Install [Balloma] (https://play.google.com/store/apps/details?id=net.blackriverstudios.balloma&hl=en) game in your android device.
    - Open Balloma and start any scene.
    - Run `python task.py`.

    It should start inputting random swipes onto the device. If scene completes for some reason (e.g the ball falls out the miniworld) it will restart automatically.

## Next steps

    - Implement object detector to extract elements in scenes that can be used to determine actual status of the game (e.g if a scene has been completed).
    - Capture scene's elements that can be used to compute the reward function (Object detector and OCR might help here).
    - Training script.
    - Tune Actor-Critic neuronal networks.
    - A lot other things that doesn't come to my mind right know.
