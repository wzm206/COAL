# COAL
## Demo of COAL framework
![img](images/show1-1.gif)
![img](images/show2-1.gif)
## Deployment

We shared our model on the [OneDrive](https://1drv.ms/f/s!Avelnwj9jiVSiZpPzkjy74ZmzXGHzA?e=6ooSSe). It is now easy to use models on robots navigation and exploration. First, download `real_env.pt` and `sample_traj.pt`, then put them into `model_weight` folder. 

If you don't have a robot yet, you can still test our framework by recording videos with your phone. There are our example video recorded through mobile phone in the folder `demo_video`. Directly run `deploy.py`:

    python deploy.py

You can achieve the following effect and you can also simply transplant this example to your own robot platform for experimentation. Green is a group of good candidate trajectories, blue is the trajectory to choose, and red is a group of bad  trajectories.

https://github.com/user-attachments/assets/0e616ff6-bb04-4a73-9509-aee557fb26e5