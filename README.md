# Deep Reinforcement Learning setup on Balloma Android game.

This work presents an actor-critic Reinforcement Learning setup, projected, implemented and executed, following the Deep Deterministic Policy Gradient approach. Two Neuronal Network architectures are presented for the actor and critic parts, being the actor a Convolutional Neuronal Network. A similar setup was applied in a 2D world game, however in this work it was applied to a 3D world game, in order to train an agent to play Balloma video game. Results demonstrate that the approach should be updated towards a policy in which exploration is favored over exploitation if cumulative reward evolution throughout training tends to decrease.

*Please read `capstone_report.pdf` for the insigts on what has been developed in this repository.

## What's been used here:

   - Minicap.
   - Keras.
   - Deep Determinist Policy Gradients.
   - OpenCV

## I want to see something running:
   - Install python 3.7.5.
   - Activate developer mode in your android device.
   - Plug android device to PC and make sure it is usable by adb `adb devices` shows online device.
   - Install lib dependencies `pip install -r requirements.txt`.
   - Clone [minicap](https://github.com/openstf/minicap#usage)
   - Run minicap with `./run.sh autosize` (You need ndk-build for this)
   - Forward requests to minicap with `adb forward tcp:1313 localabstract:minicap`
   - Install [Balloma](https://play.google.com/store/apps/details?id=net.blackriverstudios.balloma&hl=en) game in your android device.
   - Open Balloma and start game's first scene.
   - Run `python training.py`.

   It will start inputting actions onto the device infered by the Actor's ConvNet. Scene is restarted atumatically after every episode ends. Also you can use the scripts in `plots` folder of this repo to see training progress through metrics.
