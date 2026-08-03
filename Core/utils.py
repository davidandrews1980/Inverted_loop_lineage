import json
import numpy as np

def save_weights(weights, path):
    np.savez(path, **weights)

def load_weights(path):
    return dict(np.load(path))

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def human_policy(t, pos, vel, force=5.0):
    return force

def divergence_metric(agent_states, human_states):
    return np.mean(np.abs(agent_states[:, 0] - human_states[:, 0]))
