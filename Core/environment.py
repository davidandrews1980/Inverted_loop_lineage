import numpy as np

class ChaoticGravityEnvironment:
    def __init__(self, steps=500, dt=0.02):
        self.steps = steps
        self.dt = dt
        self.times = np.linspace(0, steps * dt, steps)

    def reset(self, flip_time=None, gravity_base=9.8):
        self.flip_time = flip_time if flip_time is not None else np.random.uniform(2.0, 8.0)
        self.gravity_base = gravity_base
        self.pos = 0.0
        self.vel = 0.0
        self.gravity = gravity_base
        return self.times

    def step(self, force_policy):
        states = []
        pos, vel, gravity = self.pos, self.vel, self.gravity
        for t in self.times:
            if t >= self.flip_time:
                gravity = -self.gravity_base
            if callable(force_policy):
                force = force_policy(t, pos, vel)
            else:
                idx = int(t / self.dt)
                force = force_policy[idx] if idx < len(force_policy) else 0.0
            accel = force - gravity
            vel += accel * self.dt
            pos += vel * self.dt
            if pos < -5.0 or pos > 5.0:
                vel = -vel * 0.7
                pos = np.clip(pos, -5.0, 5.0)
            states.append((pos, vel, force, gravity))
        return np.array(states)
