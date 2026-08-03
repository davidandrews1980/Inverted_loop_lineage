import os, sys, glob
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
from core.environment import ChaoticGravityEnvironment
from core.pinn import TinyPINN

def evaluate_all(lineage_path='../lineage', test_flip=5.0):
    gen_dirs = sorted(glob.glob(os.path.join(lineage_path, 'gravitas-*')))
    if not gen_dirs:
        print("No generations found.")
        return

    env = ChaoticGravityEnvironment()
    times = env.reset(flip_time=test_flip)
    human_states = env.step(lambda t, p, v: 5.0)

    plt.figure(figsize=(12, 6))
    for gen_path in gen_dirs:
        name = os.path.basename(gen_path)
        weights_path = os.path.join(gen_path, 'weights.npz')
        if not os.path.exists(weights_path):
            print(f"Skipping {name}: weights not found")
            continue
        pinn = TinyPINN.load(weights_path)
        states = env.step(lambda t, p, v: pinn.predict(t, p, v))
        plt.plot(times, states[:, 0], label=name)

    plt.plot(times, human_states[:, 0], 'k--', label='Human', linewidth=2)
    plt.axvline(test_flip, color='gray', linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Lineage Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_all()
