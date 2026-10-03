# ESE 561 Final Project Options

Oct 3, 2026 · @Anton

Four project options for the final project (30% of the grade). Each connects to a part of the course and has a working library implementation to start from; GPU access is provided.

## At a glance

| Project | Course topic | Method | Network | Compute | Difficulty |
| --- | --- | --- | --- | --- | --- |
| 1. DQN on Atari | Q-learning, function approximation | DQN (replay buffer, target network) | CNN on 4 stacked 84×84 frames | 1 GPU, several hours per run | Medium |
| 2. Kung-Fu with memory | POMDPs, recurrent policies | Actor-critic (A2C or PPO) with LSTM | CNN + LSTM | 1 GPU, several hours per run | Hard |
| 3. Ant locomotion | Continuous control, policy gradients | SAC or PPO vs. random search (ARS) | MLPs (2×256) vs. linear policy | CPU or 1 GPU, hours per run | Medium |
| 4. LLM agent | Agentic AI, POMDPs | ReAct-style tool-using agent | Pretrained LLM via API | API calls, no training | Medium |

## Project 1: DQN on Atari

Train an agent to play Breakout (or another Atari game) from raw pixels, reproducing the core of DeepMind's DQN.

**Idea.** The Q-function from the course becomes a CNN that reads the screen. Two tricks make training stable: an experience replay buffer and a slowly updated target network.

**Deliverables**

1. A working DQN on `BreakoutDeterministic-v4` or `ALE/Breakout-v5`, with a learning curve (episode return vs. frames).
2. Ablations: remove the target network, then the replay buffer; compare learning curves.
3. A gameplay video of the trained agent, and a 4-page report.

**Starting points**

- [Practical RL, week 4 DQN homework](https://github.com/yandexdataschool/Practical_RL/blob/master/week04_approx_rl/homework_pytorch_main.ipynb): a step-by-step PyTorch notebook (environment wrappers, CNN, replay buffer, target network).
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) DQN and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for tuned hyperparameters and a reference result.
- [Arcade Learning Environment docs](https://ale.farama.org/): the Atari games for Gymnasium (`pip install ale-py`).

**Visualizations to compare against**

- [Pretrained SB3 DQN on Breakout](https://huggingface.co/sb3/dqn-BreakoutNoFrameskip-v4): replay video; mean reward about 359.
- Paper: Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), Nature, 2015.

## Project 2: Kung-Fu Master with memory (POMDP)

Train an actor-critic agent with recurrent memory to play Kung-Fu Master, a game where a single frame does not show everything the agent needs.

**Idea.** One screen hides velocities and enemies that are about to appear, so the problem is partially observed. Instead of stacking frames, the agent carries an LSTM hidden state: a learned memory of the history, the same role belief states played in the search part of the course.

**Deliverables**

1. A recurrent actor-critic (CNN + LSTM) on `KungFuMasterDeterministic-v0` or `ALE/KungFuMaster-v5`, trained on parallel game copies.
2. Comparison with a memoryless agent: one frame, and 4 stacked frames, without an LSTM.
3. A small sanity check first: CartPole with velocities hidden, where a memoryless policy fails and memory fixes it.
4. A gameplay video and a 4-page report.

**Starting points**

- [Practical RL, week 8 notebook](https://github.com/yandexdataschool/Practical_RL/blob/master/week08_pomdp/practice_pytorch.ipynb): a recurrent actor-critic for Kung-Fu in PyTorch (CNN + `LSTMCell`, parallel games).
- [CleanRL `ppo_atari_lstm.py`](https://docs.cleanrl.dev/rl-algorithms/ppo/): a single-file PPO with LSTM for Atari, without stacked frames.
- [SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html): a ready-made LSTM policy for comparison.
- [Kung-Fu Master environment page](https://ale.farama.org/environments/kung_fu_master/): 14 discrete actions.

**Visualizations to compare against**

- [CleanRL PPO-LSTM Atari report](https://wandb.ai/openrlbenchmark/openrlbenchmark/reports/Atari-CleanRL-s-PPO-LSTM--VmlldzoxODcxMzE4): tracked learning curves and gameplay videos.
- Paper: Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (A3C), 2016.

## Project 3: Ant locomotion (and the maze, as a stretch goal)

Teach the four-legged MuJoCo Ant to walk, and test when a neural network is actually needed.

**Idea.** Actions are 8 continuous joint torques, so the softmax policy becomes a Gaussian policy. The project compares two very different answers: deep actor-critic (SAC, with MLP actor and critic) and Augmented Random Search, which trains a linear policy by perturbing its weights, like the parameter-space CEM from class.

**Deliverables**

1. SAC (or PPO) on `Ant-v5`: learning curve of return vs. environment steps.
2. ARS with a linear policy on the same task: learning curve vs. steps and vs. wall-clock time.
3. A discussion of sample efficiency against compute, and videos of both gaits.
4. Stretch goal: `AntMaze_UMaze-v5`, where the ant must walk to a goal through a maze; compare the sparse reward with the dense version (`AntMaze_UMazeDense-v5`), and try hindsight experience replay (HER).

**Starting points**

- [Gymnasium Ant environment](https://gymnasium.farama.org/environments/mujoco/ant/): animation, observation (105 values by default) and reward definitions.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) SAC, PPO and HER; [SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/) for ARS.
- [Gymnasium-Robotics AntMaze](https://robotics.farama.org/envs/maze/ant_maze/) for the stretch goal.

**Visualizations to compare against**

- [Pretrained SB3 SAC on Ant](https://huggingface.co/sb3/sac-Ant-v3): replay video; mean reward about 5180 (older `Ant-v3` version of the task).
- Paper: Mania, Guy and Recht, [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055), 2018.

## Project 4: A tool-using LLM agent with LangChain

Build an agent that answers questions it cannot answer alone, by deciding which tools to call, and measure where it fails.

**Idea.** The agent runs a loop: think, call a tool, read the result, repeat (the ReAct pattern). In course terms it is a POMDP agent: the LLM never sees the world directly, only tool outputs, and its context window is its memory. No training is needed; the work is in tool design, evaluation and failure analysis.

**Deliverables**

1. An agent with at least three tools, for example a calculator, a search over course notes or another document collection, and a Python or SQL tool.
2. A test set of 30 or more questions with known answers, and the agent's accuracy on it.
3. A failure analysis from the agent's traces: wrong tool, wrong arguments, loops, made-up answers. Then one fix and its measured effect.
4. A demo and a 4-page report.

**Starting points**

- [LangChain agents documentation](https://docs.langchain.com/oss/python/langchain/agents): `create_agent(model=..., tools=...)` builds the agent in a few lines.
- [LangSmith observability](https://docs.langchain.com/langsmith/observability): records every step of the agent loop (thoughts, tool calls, results), which is the visualization for this project.
- Paper: Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022.

**Cost.** The model is used through an API; set a spending limit per student. A small open-weights model on the course GPU is an alternative.

## Common requirements

- Library code is allowed, but every report explains the algorithm in the course's notation and reports at least one experiment the library does not run by default (an ablation, a comparison, a new environment).
- Learning curves average at least 3 random seeds, with the spread shown.
- Report the compute used: GPU hours or API cost.
- Final presentation: 15 minutes, including a video or a live trace of the agent.

Deadline for the proposal November 9.
