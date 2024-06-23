
# VehiclePlatooningLeon

## Generate Training Data
To generate training data, run:

```bash
python DDPG_1_Car_PID_Throttle.py
```

This will create 22 data files in the format `route.txt`. Modify the robot's movements in the Python script to produce different training datasets. Currently, the training data consists of:

- The vehicle driving straight until it reaches a certain speed.
- While maintaining its speed, the vehicle randomly decides whether to turn. If turning, it randomly chooses to turn left or right.
- After turning for a few steps, the vehicle will turn in the opposite direction for the same number of steps.
- After performing maneuvers 2 and 3, the vehicle repeats the process until reaching a total of 600 steps.

## Training Process
To train the Donkey Car, run:

```bash
python DDPG_2_Car_Ready_test.py --train
```

Press `Ctrl + C` to stop the training and wait for the training graph to be generated.

## Testing Process
Create a `route_test_1.txt` file that defines the route for the Donkey Car to follow. Then, run:

```bash
python DDPG_2_Car_Ready_test.py
```

to test the Donkey Car.

SEMANGAT IRMAN, DIMAS, DAN MARIT!!!
