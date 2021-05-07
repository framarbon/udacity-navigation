## Project Details

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

In order to install all the needed deepndencies run the following command in your virtual environment:
`pip3 -q install ./python`

You will also need to download the Unity [environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip). Extract the content into the root folder of this repo.

## Instructions

In order to start you have to open the file `Report.ipynb` in a Jupyter Notebook.
