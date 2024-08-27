import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class TrajEncoder(nn.Module):
    def __init__(self, waypoint_dim=16, dropout=0.1):
        super().__init__()
        self.projection1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3)
        self.projection2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.projection3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.gelu(self.projection1(x))
        x = self.gelu(self.projection2(x))
        x = self.pool2(x)
        x = self.gelu(self.projection3(x))
        x = self.pool3(x).squeeze()
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



class COAL_Model(nn.Module):
    def __init__(
        self,
        temperature=1,
        image_embedding=1000,
        traj_embedding=256,
        traj_error = 0.3,
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.image_encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.traj_encoder = TrajEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.traj_projection = ProjectionHead(embedding_dim=128)
        self.temperature = temperature
        self.traj_error = traj_error
        
    def get_targets(self, waypoint_ori, config):
        threshold = float(config["threshold"])
        ori_traj = waypoint_ori.clone().detach()
        batch_size, length, chanel = ori_traj.shape
        clip_length = length//2
        ori_traj = ori_traj[:, clip_length:, 1]
        label_matrix = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            now_traj = ori_traj[i]
            # 有1的行应该排除 
            dddd = torch.gt(ori_traj, now_traj-threshold) & torch.lt(ori_traj, now_traj+threshold)
            index = torch.any(dddd, dim=1)
            label_matrix[i]=index
        diag = torch.diag(label_matrix)
        
        return label_matrix - torch.diag_embed(diag)
            

    def forward(self, batch, targets):
        pass
    
    
    def get_score_deploy(self, sigle_img, traj_data):
        if sigle_img.shape[0] != 1:
            sigle_img = sigle_img.unsqueeze(0)
        # Getting Image and Action Features
        image_features = self.image_encoder(sigle_img)
        traj_features = self.traj_encoder(traj_data)
        # Getting Image and Action Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        traj_embeddings = self.traj_projection(traj_features)
        # Calculating the Loss
        logits = (traj_embeddings @ image_embeddings.T) / self.temperature
        return logits.squeeze()