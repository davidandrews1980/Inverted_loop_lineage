import os, sys, glob
sys.path.append(os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
from core.environment import ChaoticGravityEnvironment
from core.pinn import TinyPINN

def plot_lineage_comparison(lineage_path='../lineage', test_flip=5.0, save_path=None):
    gen_dirs = sorted(glob.glob(os.path.join(lineage_path, 'gravitas-*')))
    if not gen_dirs:
        print("No generations found.")
        return

    env = ChaoticGravityEnvironment()
    times = env.reset(flip_time=test_flip)
    human_states = env.step(lambda t, p, v: 5.0)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gen_dirs)))

    for i, gen_path in enumerate(gen_dirs):
        name = os.path.basename(gen_path)
        weights_path = os.path.join(gen_path, 'weights.npz')
        if not os.path.exists(weights_path):
            print(f"Skipping {name}: weights not found")
            continue
        pinn = TinyPINN.load(weights_path)
        states = env.step(lambda t, p, v: pinn.predict(t, p, v))
        plt.plot(times, states[:, 0], label=name, color=colors[i], linewidth=2)

    plt.plot(times, human_states[:, 0], 'k--', label='Human', linewidth=3, alpha=0.7)
    plt.axvline(test_flip, color='gray', linestyle=':', linewidth=2, alpha=0.6)

    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Lineage Comparison – Gravitas Generations')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    plot_lineage_comparison()
