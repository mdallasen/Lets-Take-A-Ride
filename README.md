# 🚖 Let's Take A Ride  
*Autonomous Taxi Navigation with Deep RL*  
*By Matthew Dall'Asen, Arpit Dang, Varun Satheesh*  

---

## 📖 Overview  

The objective of this report is to develop an autonomous taxi agent capable of navigating between randomly selected start and end points within New York City. Traditional approaches have typically relied on rule-based algorithms for pathfinding, which often falter in dynamic or unpredictable urban environments. In contrast, this research aims to enhance the robustness and adaptability of autonomous navigation by leveraging Reinforcement Learning (RL).

By framing navigation as a sequential decision-making problem, we train an agent to optimize its route while balancing efficiency, safety, and adaptability. Using a simulated urban environment, the agent learns through interaction to make decisions that account for real-world complexity offering a significant advantage over static methods.

This report employs Deep Q-Networks (DQN), a model-free RL algorithm that enables scalable learning and effective approximation of optimal action-value functions. Moreover, the task is structured as goal-conditioned reinforcement learning, requiring the agent to condition its policy not only on its current state but also on a specified destination. This formulation allows for better generalization across a wide range of goals and environments, moving closer to the deployment of intelligent, self-navigating agents in real-world urban settings.


---

## 📊 Evaluation Summary  

| Metric                        | Value               |
|-------------------------------|---------------------|
| **Success Rate**              | 16/20 (80%)         |
| **Average Reward**            | 46.0                |
| **Avg. Distance Traveled**    | 0.0138              |
| **Avg. Steps per Episode**    | 16.3                |
| **Detour Ratio**              | 6.15                |
| **Step Efficiency**           | 0.0010 units/step   |
| **% Toward Goal**             | 67.8%               |

*Trained for 200 episodes, tested on 20 episodes*

---

## 🌆 Environment Details  
**Core Specifications:**
- **Location:** 300m radius around NYC (40.7780, -73.9580)
- **Graph:** ~50 nodes (simplified from original 600)
- **Action Space:** Discrete (max 4 directions per node)
- **State Encoding:** Node indices + goal coordinates
- **Constraints:** One-way streets, episode limit (50 steps)

**Reward Structure:**
- `+10` for reaching goal
- `-1` per step
- `-0.5` for moving away from goal
- `-2` for repeated states

---

## 🧠 Learning Criteria & Framework

- Goal-conditioned, model-free RL agent
- Deep Q-Network (DQN) with linear and MLP layers
- State space: embedded vector of current node and goal
- Discrete action space: up to 4 valid neighboring nodes
- Epsilon-greedy exploration strategy
- Q-learning with target network for stability
- Experience replay buffer for batch updates
- Reward shaping based on:
  - Positive reward for goal completion
  - Negative penalties for revisits and suboptimal paths
  - Distance-based shaping rewards
- Episodes capped at 50 steps to prevent loops

## 📁 Project Structure

```plaintext

lets-take-a-ride/
├── data/                       # Cached OpenStreetMap data
│   └── *.graphml               # OSMnx preprocessed map files (shown as hashed filenames in screenshot)
├── report/                     # Project report
│   ├── report.pdf              # Final report
│   └── poster.jpg              # Project poster or visualization
├── src/
│   ├── envs/                   # Custom RL environment
│   │   ├── __pycache__/        # Python cache
│   │   ├── env.py              # Gymnasium environment logic
│   │   └── env_data.py         # Preprocessing and environment utilities
│   ├── execute/                # Main training and evaluation scripts
│   │   ├── __pycache__/        # Python cache
│   │   ├── train.py            # Training the agent
│   │   ├── test_model.py       # Evaluate a saved model
│   │   └── inference.py        # Run inference / deployment
│   ├── model/                  # RL algorithms
│   │   ├── __pycache__/        # Python cache
│   │   ├── DQN.py              # DQN implementation
│   │   └── ActorCritic.py      # Actor-Critic model
│   └── utils/                  # Helper functions
│       ├── __pycache__/        # Python cache
│       └── helper.py
├── main.py                     # Entry point to train and evaluate the agent
├── requirements.txt            # Python dependencies
├── trips/                      # GIFs of agent trajectories
├── .gitignore                  # git ignore for data and other folders/files
├── LICENSE                     # License file
└── README.md                   # Project documentation

```

---

## ▶️ How to Use

Follow these steps to get started:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/lets-take-a-ride.git
    cd lets-take-a-ride
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the main script**:
    ```bash
    python main.py
    ```

---

