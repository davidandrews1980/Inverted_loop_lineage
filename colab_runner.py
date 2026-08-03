# colab_runner.py – the one‑stop script for Colab or local runs
# Run this in a Colab cell or with `python colab_runner.py`

import os
import sys
import subprocess

# Clone the repository if not already present
if not os.path.exists('Inverted_loop_lineage'):
    os.system('git clone https://github.com/davidandrews1980/Inverted_loop_lineage.git')
    os.chdir('Inverted_loop_lineage')
else:
    os.chdir('Inverted_loop_lineage')

# Ensure the lineage directory exists
os.makedirs('lineage/gravitas-0', exist_ok=True)

# Add core to path
sys.path.append('.')

# Import required modules
from core.pinn import TinyPINN
from core.environment import ChaoticGravityEnvironment
import numpy as np
import json

print("=== Seeding Gravitas‑0 ===")
# Create a random initial Gravitas‑0 (or load existing if present)
if not os.path.exists('lineage/gravitas-0/weights.npz'):
    pinn0 = TinyPINN()
    pinn0.save('lineage/gravitas-0/weights.npz')
    with open('lineage/gravitas-0/config.json', 'w') as f:
        json.dump({'epochs': 0, 'human_force': 5.0, 'branch': 'main', 'parent': 'none'}, f, indent=2)
    print("Gravitas‑0 created.")
else:
    print("Gravitas‑0 already exists.")

print("\n=== Training Gravitas‑1 from Gravitas‑0 ===")
from scripts.train_next import train_next
train_next(
    parent_path='lineage/gravitas-0/weights.npz',
    child_name='gravitas-1',
    epochs=30,
    human_force=5.0,
    branch='main'
)

print("\n=== Evaluating all generations ===")
from scripts.evaluate_all import evaluate_all
evaluate_all(lineage_path='lineage')

print("\n=== Lineage plot ===")
from scripts.lineage_plot import plot_lineage_comparison
plot_lineage_comparison(lineage_path='lineage', save_path='lineage_plot.png')

print("\nDone! Check lineage_plot.png for the comparison.")
