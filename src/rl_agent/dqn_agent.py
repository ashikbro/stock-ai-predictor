"""
Deep Q-Network (DQN) agent for stock trading.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


class DQN(nn.Module):
    """Deep Q-Network neural network."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize DQN.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class ReplayMemory:
    """Experience replay memory."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """DQN Agent for stock trading."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        learning_rate: float = None,
        gamma: float = None,
        epsilon_start: float = None,
        epsilon_end: float = None,
        epsilon_decay: float = None,
        memory_size: int = None,
        batch_size: int = None,
        device: str = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            memory_size: Replay memory capacity
            batch_size: Batch size for training
            device: Device to use (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.gamma = gamma or config.GAMMA
        self.epsilon = epsilon_start or config.EPSILON_START
        self.epsilon_end = epsilon_end or config.EPSILON_END
        self.epsilon_decay = epsilon_decay or config.EPSILON_DECAY
        self.batch_size = batch_size or config.BATCH_SIZE
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size or config.MEMORY_SIZE)
        
        # Training stats
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            evaluate: If True, use greedy policy (no exploration)
            
        Returns:
            Selected action
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")
