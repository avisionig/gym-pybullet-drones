import pybullet as p

p.connect(p.GUI)
try:
    spaceship_id = p.loadURDF("basic_spaceship.urdf", basePosition=[0, 0, 0])
    print("Spaceship loaded successfully:", spaceship_id)
except Exception as e:
    print("Error loading URDF:", e)
p.disconnect()
