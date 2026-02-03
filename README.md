# active-perception-RL-navigation
This repository contains the source code for the paper: [_**Reinforcement Learning for Active Perception in Autonomous Navigation**_](https://arxiv.org/abs/2602.01266).

A video demonstrating the work is available [here](https://www.youtube.com/@autonomousrobotslab).


## 🛠️ Installation
1. Install Isaac Gym and Aerial Gym Simulator

   Follow the [instructions](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/#installation )  provided in the respective repository.
   > ### ⚠️ Important Note: Change to Argument Parser in Isaac Gym's `gymutil.py`
   >
   > Before installing the Aerial Gym Simulator, you must modify the Isaac Gym installation.
   > The argument parser in Isaac Gym may interfere with additional arguments required by other learning frameworks. To resolve this, you need to modify line 337 of the `gymutil.py` file located in the `isaacgym` folder.
   >
   > Change the following line:
   > 
   > ```python
   > args = parser.parse_args()
   > ```
   >
   > to:
   >
   > ```python
   > args, _ = parser.parse_known_args()
   > ```

2. Set up the environment

   Once the installation is successful, activate the `aerialgym` environment:
   ```bash
   cd ~/workspaces/ && conda activate aerialgym
   ```
3. Clone this repository

   Clone the repository by running the following command:
   ```bash
   git clone git@github.com:ntnu-arl/active-perception-RL-navigation.git
   ```
4. Install active-perception-RL-navigation

   Navigate to the cloned repository and install it using the following command:
   ```bash
   cd ~/workspaces/active-perception-RL-navigation/
   pip install -e .
   ```

## 🚀 Running the Examples

The standalone examples, along with a pre-trained RL policy, are available in the [`examples`](./examples) directory.  
The ready-to-use policy (used in the work described in [_**Reinforcement Learning for Active Perception in Autonomous Navigation**_](-)) can be found under: `examples/pre-trained_network`. These examples illustrate policy inference in a corridor-like environment under different levels of complexity, specifically with 10, 20, and 30 obstacles.

### 10-Obstacle Example

Run the following:
```bash
cd ~/workspaces/active-perception-RL-navigation/examples/
conda activate aerialgym
bash example_10obstacles.sh
```
You should now be able to observe the trained policy in action — performing a navigation task with actively actuated camera sensor in the environment:


https://github.com/user-attachments/assets/d0894097-a514-4932-af57-fffdff8fb721


### 20-Obstacle Example
Run the following:
```bash
cd ~/workspaces/active-perception-RL-navigation/examples/
conda activate aerialgym
bash example_20obstacles.sh
```
🎥 Demo:


https://github.com/user-attachments/assets/97fb85a8-afbd-40ea-b548-3eb6b70257e6


### 30-Obstacle Example
Run the following:
```bash
cd ~/workspaces/active-perception-RL-navigation/examples/
conda activate aerialgym
bash example_30obstacles.sh
```
🎥 Demo:


https://github.com/user-attachments/assets/151fc99a-9f91-4ad5-bf94-ab2e0fae8fa2



## 🏋️ RL Training
### Running Training
To train your first **active perception RL navigation policy**, run:
```bash
conda activate aerialgym
cd ~/workspaces/active-perception-RL-navigation/
python -m rl_training.train_aerialgym --env=navigation_active_camera_task --experiment=testExperiment
```
By default, the number of environments is set to 1024. If your GPU cannot handle this load, reduce it by adjusting the `num_envs` parameter in `/src/config/task/navigation_active_camera_task_config.py`:

```python
num_envs = 1024
```
By default, the training environment contains 38 obstacles. You can modify this by editing the `num_assets` parameter in `/src/config/assets/env_object_config.py`:
```python
class object_asset_params(asset_state_params):
    num_assets = 35
```
and
```python
class panel_asset_params(asset_state_params):
    num_assets = 3
```


### Loading Trained Models
To load a trained checkpoint and perform inference only (no training), follow these steps:

1. For clear visualization (to avoid rendering overhead), reduce the number of environments (e.g., to 16) and enable the viewer by modifying `/src/config/task/navigation_active_camera_task_config.py`:

   From:
   ```python
   num_envs = 512
   use_warp = True
   headless = True
   ```
   To:
   ```python
   num_envs = 16
   use_warp = True
   headless = False
   ```
2. For a better view during inference, consider excluding the top wall from the corridor-like environments by modifying the `/src/config/env/env_active_camera_with_obstacles.py` file:

   ```python
   "top_wall": False, # excluding top wall
   ```
3. Finally, execute the inference script with the following command:

   ```bash
   conda activate aerialgym
   cd ~/workspaces/active-perception-RL-navigation/
   python -m rl_training.enjoy_aerialgym --env=navigation_active_camera_task --experiment=testExperiment
   ```
   The default viewer is set to follow the agent. To disable this feature and inspect other parts of the environment, press `F` on your keyboard.


## 📄 Citing

If you use or reference this work in your research, please cite the following paper:

G. Malczyk, M. Kulkarni and K. Alexis, "Reinforcement Learning for Active Perception in Autonomous Navigation", 2025

```bibtex
@article{malczyk2025reinforcement,
  title={Reinforcement Learning for Active Perception in Autonomous Navigation},
  author={Malczyk, Grzegorz and Kulkarni, Mihir and Alexis, Kostas},
  journal={arXiv preprint arXiv:2602.01266},
  year={2026}
}
```

## Contact
For inquiries, feel free to reach out to the authors:
- **Grzegorz Malczyk**
  
  [Email](mailto:grzegorz.malczyk@ntnu.no) | [GitHub](https://github.com/grzemal) | [LinkedIn](https://www.linkedin.com/in/grzegorz-malczyk/) | [X (formerly Twitter)](https://twitter.com/grzemalige)

- **Mihir Kulkarni**

  [Email](mailto:mihirk284@gmail.com) | [GitHub](https://github.com/mihirk284) | [LinkedIn](https://www.linkedin.com/in/mihir-kulkarni-6070b6135/) | [X (formerly Twitter)](https://twitter.com/mihirk284)

- **Kostas Alexis**

  [Email](mailto:konstantinos.alexis@ntnu.no) | [GitHub](https://github.com/kostas-alexis) | [LinkedIn](https://www.linkedin.com/in/kostas-alexis-67713918/) | [X (formerly Twitter)](https://twitter.com/arlteam)

This research was conducted at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com/), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no). 

For more information, visit our website.

## Acknowledgements
This material was supported by the Research Council of Norway under Award NO-338694 and the Horizon Europe Grant Agreement No. 101119774.

Additionally, this repository incorporates code and helper scripts from the [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator).



![arl_ntnu_logo_v2](https://github.com/user-attachments/assets/f4208309-d0a4-4084-b5aa-14adf4cb7e6c)
