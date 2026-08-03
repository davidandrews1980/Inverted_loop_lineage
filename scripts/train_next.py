import os, sys, json
import numpy as np
sys.path.append(os.path.abspath('..'))
from core.environment import ChaoticGravityEnvironment
from core.pinn import TinyPINN

def train_next(parent_path, child_name, epochs=30, human_force=5.0, branch='main'):
    parent = TinyPINN.load(parent_path)
    child = TinyPINN(weights=parent.get_weights())
    env = ChaoticGravityEnvironment()

    print(f"Training {child_name} (branch: {branch})...")
    for epoch in range(epochs):
        flip_time = np.random.uniform(2.0, 8.0)
        times = env.reset(flip_time=flip_time)
        human_states = env.step(lambda t, p, v: human_force)

        sample_idx = np.linspace(0, len(times)-1, len(times)//10, dtype=int)
        t_samples = times[sample_idx]
        pos_samples = human_states[sample_idx, 0]
        vel_samples = human_states[sample_idx, 1]

        agent_forces = np.array([child.predict(t_samples[i], pos_samples[i], vel_samples[i])
                                 for i in range(len(t_samples))])
        phys_loss = np.mean((agent_forces - human_force)**2)
        anti_loss = -np.mean((agent_forces - human_force)**2)
        loss = phys_loss + 0.5 * anti_loss

        for param in [child.W1, child.b1, child.W2, child.b2]:
            param += np.random.randn(*param.shape) * 0.001

        if epoch % 10 == 0:
            print(f"{epoch}: loss={loss:.4f}, phys={phys_loss:.4f}, anti={-anti_loss:.4f}")

    # Save child
    child_dir = os.path.join('../lineage', child_name)
    os.makedirs(child_dir, exist_ok=True)
    child.save(os.path.join(child_dir, 'weights.npz'))
    with open(os.path.join(child_dir, 'config.json'), 'w') as f:
        json.dump({
            'parent': parent_path,
            'epochs': epochs,
            'human_force': human_force,
            'branch': branch
        }, f, indent=2)
    print(f"{child_name} saved.")

if __name__ == "__main__":
    # Example usage: train next generation from the latest
    # You'll call this from the notebook or command line.
    pass
