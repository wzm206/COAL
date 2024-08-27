import cv2
import torch
from torchvision import transforms
from model.COAL import COAL_Model
import numpy as np

device = "cuda"
video_path = "demo_video/road.mp4"
model = COAL_Model().to(device)
checkpoint = torch.load("model_weight/real_env.pt", map_location='cuda:0')
model.load_state_dict(checkpoint)
model.eval()

to_tensor = transforms.ToTensor()

# Action sequence candidates
traj_dic = torch.load("model_weight/sample_traj.pt")
waypoint = traj_dic["waypoint_ori_train"].to(device)
waypoint_normal_train = traj_dic["waypoint_normal_train"].to(device)
batch_data = {}
batch_data["traj"] = waypoint_normal_train

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

cap = cv2.VideoCapture(video_path)     
while cap.isOpened():               
    ret, frame = cap.read()
    if ret:
        # 720, 1280, 3
        h, w, c = frame.shape
        # 600*450
        new_frame = frame[:450, w//2-300:w//2+300]
        image_now = cv2.resize(new_frame, (224, 224))
        obs_images = to_tensor(image_now).to(device)
        batch_data["image"] = obs_images
        batch_score = model.get_score_deploy(obs_images, waypoint_normal_train)

        _, top5_index = torch.topk(batch_score, k=5, dim=0, largest=True, sorted=True)
        _, last5_index = torch.topk(batch_score, k=5, dim=0, largest=False, sorted=True)

        top5_data = to_numpy(torch.index_select(waypoint, dim=0, index=top5_index))
        last5_data = to_numpy(torch.index_select(waypoint, dim=0, index=last5_index))
        background = np.full((720, 500, 3), 255, dtype=np.uint8)
        
        waypoints_int = (50*top5_data).astype(int)
        waypoints_int = waypoints_int[:, :, [1,0]]
        waypoints_int[:, :, 0] = -1*waypoints_int[:, :, 0] + 500//2
        waypoints_int[:, :, 1] = 720-waypoints_int[:, :, 1]
        cv2.polylines(background, waypoints_int, isClosed=False, color=(0,255,0), thickness=2)
        cv2.polylines(background, waypoints_int[:1], isClosed=False, color=(255,0,0), thickness=2)
        waypoints_int = (50*last5_data).astype(int)
        waypoints_int = waypoints_int[:, :, [1,0]]
        waypoints_int[:, :, 0] = -1*waypoints_int[:, :, 0] + 500//2
        waypoints_int[:, :, 1] = 720-waypoints_int[:, :, 1]
        cv2.polylines(background, waypoints_int, isClosed=False, color=(0,0,255), thickness=1)

        full_image = np.concatenate([frame, background], axis=1)
        cv2.imshow('frame', full_image)
        key = cv2.waitKey(20)       
        if key == ord('q'):         
            cap.release()
            break
    else:
        cap.release()
cv2.destroyAllWindows()