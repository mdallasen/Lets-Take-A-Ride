# 🚖 Let's Take A Ride  
*Autonomous Taxi Navigation with Deep RL*  
*By Matthew Dall'Asen, Arpit Dang, Varun Satheesh*  

---

## 📖 Overview  
A reinforcement learning agent that learns to navigate a real-world inspired New York City street network using Goal-Conditioned Deep Q-Learning. The environment is built from OpenStreetMap data with realistic constraints like one-way streets and dynamic action spaces.

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
│   ├── *.graphml               # OSMnx preprocessed map files
├── src/
│   ├── envs/                   # Custom RL environment
│   │   ├── env.py              # Gymnasium environment logic
│   │   └── env_data.py         # Preprocessing and environment utilities
│   ├── execute/                # Main training and evaluation scripts
│   │   ├── train.py            # Training the agent
│   │   ├── test_model.py       # Evaluate a saved model
│   │   └── inference.py        # Run inference / deployment
│   ├── model/                  # DQN and other RL algorithms
│   │   ├── DQN.py              # Deep Q-Network implementation
│   │   └── ActorCritic.py      # Optional: Actor-Critic model
│   └── utils/                  # Helper functions
│       └── helper.py
├── main.py                     # Entry point to train and evaluate the agent
├── requirements.txt            # Python dependencies
├── trips/                      # GIFs of agent trajectories
├── .gitignore                  # git ignore for data and other folders / files
├── poster.jpg                  # Project poster or visualization
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

