import torch
import torch.nn as nn
import streamlit as st

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class Conv2D_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            padding=0, dilation=1, activation="relu"
    ):
        super(Conv2D_activa, self).__init__()
        self.padding = padding
        if self.padding:
            self.pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, bias=None
        )
        self.activation = activation
        if activation == "relu":
            self.activation = nn.ReLU()

    def forward(self, x):
        if self.padding:
            x = self.pad(x)
        x = self.conv2d(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, dim_intermediate=32, ks=3, s=1):
        super(ResBlk, self).__init__()
        p = (ks - 1) // 2
        self.cba_1 = Conv2D_activa(dim_in, dim_intermediate, ks, s, p, activation="relu")
        self.cba_2 = Conv2D_activa(dim_intermediate, dim_out, ks, s, p, activation=None)

    def forward(self, x):
        y = self.cba_1(x)
        y = self.cba_2(y)
        return y + x


def _repeat_blocks(block, dim_in, dim_out, num_blocks, dim_intermediate=32, ks=3, s=1):
    blocks = []
    for idx_block in range(num_blocks):
        if idx_block == 0:
            blocks.append(block(dim_in, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
        else:
            blocks.append(block(dim_out, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    def __init__(
            self, dim_in=3, dim_out=32, num_resblk=3,
            use_texture_conv=True, use_motion_conv=True, texture_downsample=True,
            num_resblk_texture=2, num_resblk_motion=2
    ):
        super(Encoder, self).__init__()
        self.use_texture_conv, self.use_motion_conv = use_texture_conv, use_motion_conv

        self.cba_1 = Conv2D_activa(dim_in, 16, 7, 1, 3, activation="relu")
        self.cba_2 = Conv2D_activa(16, 32, 3, 2, 1, activation="relu")

        self.resblks = _repeat_blocks(ResBlk, 32, 32, num_resblk)

        # texture representation
        if self.use_texture_conv:
            self.texture_cba = Conv2D_activa(
                32, 32, 3, (2 if texture_downsample else 1), 1,
                activation="relu"
            )
        self.texture_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_texture)

        # motion representation
        if self.use_motion_conv:
            self.motion_cba = Conv2D_activa(32, 32, 3, 1, 1, activation="relu")
        self.motion_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_motion)

    def forward(self, x):
        x = self.cba_1(x)
        x = self.cba_2(x)
        x = self.resblks(x)

        if self.use_texture_conv:
            texture = self.texture_cba(x)
            texture = self.texture_resblks(texture)
        else:
            texture = self.texture_resblks(x)

        if self.use_motion_conv:
            motion = self.motion_cba(x)
            motion = self.motion_resblks(motion)
        else:
            motion = self.motion_resblks(x)

        return texture, motion


class Decoder(nn.Module):
    def __init__(self, dim_in=32, dim_out=3, num_resblk=9, texture_downsample=True):
        super(Decoder, self).__init__()
        self.texture_downsample = texture_downsample

        if self.texture_downsample:
            self.texture_up = nn.UpsamplingNearest2d(scale_factor=2)
            # self.texture_cba = Conv2D_activa(dim_in, 32, 3, 1, 1, activation="relu")

        self.resblks = _repeat_blocks(ResBlk, 64, 64, num_resblk, dim_intermediate=64)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.cba_1 = Conv2D_activa(64, 32, 3, 1, 1, activation="relu")
        self.cba_2 = Conv2D_activa(32, dim_out, 7, 1, 3, activation=None)

    def forward(self, texture, motion):
        if self.texture_downsample:
            texture = self.texture_up(texture)
        if motion.shape != texture.shape:
            texture = nn.functional.interpolate(texture, size=motion.shape[-2:])
        x = torch.cat([texture, motion], 1)

        x = self.resblks(x)

        x = self.up(x)
        x = self.cba_1(x)
        x = self.cba_2(x)

        return x


class Manipulator(nn.Module):
    def __init__(self):
        super(Manipulator, self).__init__()
        self.g = Conv2D_activa(32, 32, 3, 1, 1, activation="relu")
        self.h_conv = Conv2D_activa(32, 32, 3, 1, 1, activation=None)
        self.h_resblk = ResBlk(32, 32)

    def forward(self, motion_A, motion_B, amp_factor):
        motion = motion_B - motion_A
        motion_delta = self.g(motion) * amp_factor
        motion_delta = self.h_conv(motion_delta)
        motion_delta = self.h_resblk(motion_delta)
        motion_mag = motion_B + motion_delta
        return motion_mag


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = Encoder(dim_in=3*1)
        self.manipulator = Manipulator()
        self.decoder = Decoder(dim_out=3*1)

    def forward(self, batch_A, batch_B, batch_C, batch_M, amp_factor, mode="train"):
        if mode == "train":
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            texture_C, motion_C = self.encoder(batch_C)
            texture_M, motion_M = self.encoder(batch_M)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            texture_AC = [texture_A, texture_C]
            motion_BC = [motion_B, motion_C]
            texture_BM = [texture_B, texture_M]
            return y_hat, texture_AC, texture_BM, motion_BC
        elif mode == "evaluate":
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            return motion_mag, y_hat
        
import os
import random
from typing import Union, Tuple

import cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_heads=2, dropout_rate=0.7):
        super(GATLayer, self).__init__()
        self.gat1 = GATConv(int(in_channels), hidden_channels, heads=num_heads, concat=True, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout_rate)
        self.gat3 = GATConv(hidden_channels * num_heads, int(out_channels), heads=num_heads, concat=False, dropout=dropout_rate)
        
        self.norm1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.norm2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.norm3 = nn.BatchNorm1d(int(out_channels))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Residual connection layer
        self.residual = nn.Linear(int(in_channels), int(out_channels))

    def forward(self, x, edge_index):
        residual = self.residual(x)
        
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.gat3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x + residual  # Adding the residual connection
        
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, y, special_matrix):
#         print(f"Special matrix shape: {special_matrix.shape}")
#         print(f"Input feature shape: {y.shape}")
        y = y @ self.weight  # (num_nodes, in_features) @ (in_features, out_features) -> (num_nodes, out_features)
#         print(f"After weight multiplication shape: {y.shape}")
        y = special_matrix @ y  # (num_nodes, num_nodes) @ (num_nodes, out_features) -> (num_nodes, out_features)
#         print(f"After special matrix multiplication shape: {y.shape}")
        if self.bias is not None:
            y += self.bias
        return y

class GCN(nn.Module):
    def __init__(self, adj_matrix, hidden_features=160, num_embeddings=100, in_features=51, out_features=320):
        super(GCN, self).__init__()
        self.adj_matrix = adj_matrix
        adj_matrix += torch.eye(adj_matrix.size(0)).to(adj_matrix.device)
        degree_matrix = torch.sum(adj_matrix != 0.0, axis=1)
        inverse_degree_sqrt = torch.diag(torch.pow(degree_matrix, -0.5))
        self.special_matrix = (adj_matrix @ inverse_degree_sqrt).transpose(0, 1) @ inverse_degree_sqrt
        self.graph_weight_one = GraphConvolution(in_features=in_features, out_features=hidden_features, bias=False)
        self.graph_weight_two = GraphConvolution(in_features=hidden_features, out_features=hidden_features, bias=False)
        self.graph_weight_three = GraphConvolution(in_features=hidden_features, out_features=out_features, bias=False)

    def forward(self, y):
        # Adjust special_matrix size to match the number of nodes
        num_nodes = self.adj_matrix.size(0)
        special_matrix = self.special_matrix[:num_nodes, :num_nodes]
        
        y = self.graph_weight_one(y, special_matrix)
        y = F.leaky_relu(y, 0.2)
        y = self.graph_weight_two(y, special_matrix)
        y = F.leaky_relu(y, 0.2)
        y = self.graph_weight_three(y, special_matrix)
        y = F.leaky_relu(y, 0.2)
        return y

class GCNEncoder(nn.Module):
    def __init__(self, adj_matrix, num_embeddings, hidden_features=160, in_features=51, out_features=320):
        super(GCNEncoder, self).__init__()
        self.gcn = GCN(adj_matrix, num_embeddings=num_embeddings, hidden_features=hidden_features, in_features=in_features, out_features=out_features)

    def forward(self, y):
        return self.gcn(y)

class VGAE_FacialGraph(nn.Module):
    def __init__(self, adj_matrix, in_channels=51, out_channels=32, latent_dim=16, num_embeddings=100, num_classes=3, hidden_features=160, in_features=51, beta=0.1, temperature=1, dropout_rate=0.3):
        super(VGAE_FacialGraph, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.temperature = temperature
        self.gat_encoder = GATLayer(in_channels=int(in_channels), out_channels=int(out_channels), dropout_rate=dropout_rate)
        self.gcn_encoder = GCNEncoder(adj_matrix, num_embeddings, hidden_features, in_features, out_channels)
        
        self.mean_gat = GATLayer(in_channels=int(out_channels), out_channels=int(latent_dim), dropout_rate=dropout_rate)
        self.log_var_gat = GATLayer(in_channels=int(out_channels), out_channels=int(latent_dim), dropout_rate=dropout_rate)
        
        self.mean_gcn = nn.Linear(int(out_channels), int(latent_dim))
        self.log_var_gcn = nn.Linear(int(out_channels), int(latent_dim))
        
        self.norm = nn.LayerNorm(int(latent_dim) * 2)
        
        self.fc = nn.Linear(int(latent_dim) * 2, int(num_classes))
        self.dropout = nn.Dropout(dropout_rate)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, x, edge_index):
        x_flat = x.view(-1, x.size(-1))  # Flatten the tensor for the GAT layers
        
        x_gat_encoded = self.gat_encoder(x_flat, edge_index)
        mean_gat = self.mean_gat(x_gat_encoded, edge_index)
        log_var_gat = self.log_var_gat(x_gat_encoded, edge_index)
        z_gat = self.reparameterize(mean_gat, log_var_gat)
        
        x_gcn_encoded = self.gcn_encoder(torch.zeros((self.gcn_encoder.gcn.adj_matrix.size(0), x.size(-1)), device=x.device))
        mean_gcn = self.mean_gcn(x_gcn_encoded)
        log_var_gcn = self.log_var_gcn(x_gcn_encoded)
        z_gcn = self.reparameterize(mean_gcn, log_var_gcn)

        # Padding Z_GCN to match the size of Z_GAT
        if z_gat.size(0) != z_gcn.size(0):
            if z_gat.size(0) > z_gcn.size(0):
                padding_size = z_gat.size(0) - z_gcn.size(0)
                z_gcn = F.pad(z_gcn, (0, 0, 0, padding_size))
            else:
                padding_size = z_gcn.size(0) - z_gat.size(0)
                z_gat = F.pad(z_gat, (0, 0, 0, padding_size))

        # Concatenate both latent representations
        z = torch.cat((z_gat, z_gcn), dim=1)
        z = F.relu(self.norm(z))

        # Inner product decoder
        adj_reconstructed = torch.sigmoid(torch.matmul(z, z.t()))

        class_probs = self.fc(z) / self.temperature

        return class_probs, adj_reconstructed, (mean_gat, log_var_gat), (mean_gcn, log_var_gcn)
    
import os
import torch
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import dlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
magnet = MagNet().to(device)
magnet.load_state_dict(torch.load(r"weight\magnet_epoch12_loss7.28e-02 (3).pth", map_location=device))  # Ensure the model is loaded

EYEBROW_INDEX = (17, 27)
MOUTH_INDEX = (48, 68)
MAX_EDGES = 1500  # Based on MEDataset

# Initialize the dlib face detector and shape predictor
PREDICTOR_PATH = r"weight\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


# Utility functions
def reset_page():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    width, height = img.shape[1], img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    if isinstance(crop_size, tuple):
        crop_width, crop_height = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_height = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_height:mid_y + crop_height, mid_x - crop_width:mid_x + crop_width]
    return crop_img

def unit_preprocessing(unit):
    # Apply center crop only if the dataset is "samm"
    if dataset_name == "samm":
        unit = center_crop(unit, (420, 420))
    unit = cv2.resize(unit, (256, 256))
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    unit = np.transpose(unit / 127.5 - 1.0, (2, 0, 1))
    unit = torch.FloatTensor(unit).unsqueeze(0)
    return unit

    # unit = cv2.resize(unit, (256, 256))
    # unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    # unit = np.transpose(unit / 127.5 - 1.0, (2, 0, 1))
    # unit = torch.FloatTensor(unit).unsqueeze(0)
    # return unit

def magnify_postprocessing(unit):
    unit = unit[0].permute(1, 2, 0).contiguous()
    unit = (unit + 1.0) * 127.5
    unit = unit.numpy().astype(np.uint8)
    unit = cv2.cvtColor(unit, cv2.COLOR_RGB2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit

def unit_postprocessing(unit):
    unit = unit[0]
    max_v = torch.amax(unit, dim=(1, 2), keepdim=True)
    min_v = torch.amin(unit, dim=(1, 2), keepdim=True)
    unit = (unit - min_v) / (max_v - min_v + 1e-8)
    unit = torch.mean(unit, dim=0).numpy()
    unit = cv2.resize(unit, (128, 128))
    return unit

def get_patches(point: Tuple[int, int], patch_size: int = 7) -> Tuple[int, int, int, int]:
    half_size = patch_size // 2
    return point[0] - half_size, point[0] + half_size + 1, point[1] - half_size, point[1] + half_size + 1

def detect_landmarks(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return []
    landmarks = predictor(gray, faces[0])
    return [(p.x, p.y) for p in landmarks.parts()]

def adjacency_to_edge_index(adj_matrix):
    rows, cols = np.where(adj_matrix != 0)
    edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
    return edge_index

def pad_edge_index(edge_index, max_edges, num_nodes):
    num_edges = edge_index.size(1)
    padded = torch.zeros((2, max_edges), dtype=torch.long)
    
    if num_edges > max_edges:
        padded = edge_index[:, :max_edges]
    else:
        padded[:, :num_edges] = edge_index
        if num_edges < max_edges:
            pad_count = max_edges - num_edges
            pad_indices = torch.randint(0, num_nodes, (2, pad_count))
            padded[:, num_edges:] = pad_indices

    # Filter out-of-bound indices
    valid_mask = (padded[0] < num_nodes) & (padded[1] < num_nodes)
    padded = padded[:, valid_mask]

    return padded

def create_knn_graph(landmarks, k=8):
    nodes = np.array(landmarks)
    edges = []
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(nodes)
    distances, indices = nbrs.kneighbors(nodes)
    
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                edges.append((i, neighbor))

    adjacency_matrix = np.zeros((len(landmarks), len(landmarks)))
    for i, neighbors in enumerate(indices):
        avg_distance = np.mean(distances[i])
        sigma = max(avg_distance, 1e-8)  # Ensure sigma is not zero
        for neighbor in neighbors:
            if i != neighbor:
                dist = np.linalg.norm(nodes[i] - nodes[neighbor])
                adjacency_matrix[i, neighbor] = np.exp(-dist**2 / (2 * sigma**2))
                adjacency_matrix[neighbor, i] = adjacency_matrix[i, neighbor]

    return nodes, edges, adjacency_to_edge_index(adjacency_matrix)

def detect_landmarks(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return []
    landmarks = predictor(gray, faces[0])
    return [(p.x, p.y) for p in landmarks.parts()]

# Load the pre-trained model and adjacency matrix
model_path = r"weight\model_best.pt"
adj_matrix_path = r"weight\4.npz"

npz_file = np.load(adj_matrix_path)
adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to(device)

model = VGAE_FacialGraph(adj_matrix=adj_matrix, num_classes=3, temperature=1, dropout_rate=0.6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = center_crop(img, (420, 420))
    img = unit_preprocessing(img)
    return img

# Detection and feature extraction
def extract_features_from_images(onset_img, apex_img):
    onset_frame = preprocess_image(onset_img)
    apex_frame = preprocess_image(apex_img)

    if onset_frame is None or apex_frame is None:
        return None, None, None

    onset_frame = onset_frame.to("cpu")
    apex_frame = apex_frame.to("cpu")

    with torch.no_grad():
        amp_factor = 2.0
        shape_representation, magnify = magnet(batch_A=onset_frame, batch_B=apex_frame, batch_C=None, batch_M=None,
                                               amp_factor=amp_factor, mode="evaluate")

    magnify = magnify_postprocessing(magnify.to("cpu"))
    shape_representation = unit_postprocessing(shape_representation.to("cpu"))

    points = detect_landmarks(magnify)
    if not points:
        return None, None, None

    points = points[EYEBROW_INDEX[0]:EYEBROW_INDEX[1]] + points[MOUTH_INDEX[0]:MOUTH_INDEX[1]]

    scaler = StandardScaler()
    normalized_landmarks = scaler.fit_transform(points)
    normalized_landmarks = np.clip(normalized_landmarks, 0, 1)
    nodes, edges, adj_matrix = create_knn_graph(normalized_landmarks, k=5)

    patches = []
    for point in points:
        start_x, end_x, start_y, end_y = get_patches(point, 7)
        patch = np.expand_dims(shape_representation[start_x:end_x, start_y:end_y], axis=-1)

        if patch.size == 0:
            continue

        patch = cv2.resize(patch, (7, 7))
        patches.append(torch.tensor(patch, dtype=torch.float))

    if not patches:
        return None, None, None

    patches = torch.stack(patches)
    nodes_tensor = torch.tensor(normalized_landmarks, dtype=torch.float)
    nodes_tensor = nodes_tensor.view(nodes_tensor.size(0), -1)
    patches = patches.view(patches.size(0), -1)

    combined_features = torch.cat([nodes_tensor, patches], dim=-1).view(len(points), -1)

    edge_index = adjacency_to_edge_index(adj_matrix)
    edge_index = pad_edge_index(edge_index, MAX_EDGES, combined_features.size(0)).to(torch.long)

    return combined_features, edge_index, points


def extract_ground_truth_from_filename(filename):
    """Extract ground truth emotion based on the last part of the filename before the extension."""
    # Split the filename and extract the last part before the file extension
    base_name = os.path.splitext(filename)[0]  # Remove the file extension
    ground_truth_emotion = base_name.split('_')[-1]  # Get the last part after splitting by '_'
    
    return ground_truth_emotion


label_mapping = {
        "Positive": 0,  
        "Surprise": 1, 
        "Negative": 2,
}

index_to_emotion = {v: k for k, v in label_mapping.items()}


def predict_emotion(onset_img_path, apex_img_path):
    combined_features, edge_index, points = extract_features_from_images(onset_img_path, apex_img_path)
    if combined_features is None:
        return None, None

    with torch.no_grad():
        output, _, _, _ = model(combined_features, edge_index)
        prediction = torch.argmax(output, dim=1)

    predicted_scalar = prediction[0].item()
    predicted_emotion = index_to_emotion.get(predicted_scalar, "Unknown")

    ground_truth_emotion = extract_ground_truth_from_filename(onset_img_path)

    return predicted_emotion, ground_truth_emotion


# Set file paths
PREDICTOR_PATH = r"weight\shape_predictor_68_face_landmarks.dat"
model_path = r"weight\model_best.pt"
adj_matrix_path = r"weight\4.npz"

# Load model and predictor
predictor = dlib.shape_predictor(PREDICTOR_PATH)
npz_file = np.load(adj_matrix_path)
adj_matrix = torch.FloatTensor(npz_file["adj_matrix"]).to("cpu")
magnet = MagNet().to("cpu")
magnet.load_state_dict(torch.load(
    "weight\magnet_epoch12_loss7.28e-02 (3).pth", map_location="cpu"))
magnet.eval()

model = VGAE_FacialGraph(adj_matrix=adj_matrix, num_classes=3, temperature=1, dropout_rate=0.6)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.to("cpu")
model.eval()

import streamlit as st
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="Micro-Expression Recognition", page_icon=":smiley:", layout="wide")

dataset_name = st.selectbox("Select Dataset", options=["samm", "other"], index=0)

# Custom CSS for improved styling and layout
st.markdown("""
    <style>
    /* Styling the Streamlit app */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    body {
        background-color: #f7f5f2;
        color: #333333;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
        max-width: 900px;
        margin: auto;
    }
    h1 {
        color: #74C6D4;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Change the background and text color of the file uploader */
    .stFileUploader label, .stFileUploader div div div span {
        background-color: #333333 !important; /* Dark background */
        color: #c0c0c0 !important; /* Pastel grey text */
        padding: 10px;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: #FFCCD5;
        color: #c0c0c0; /* Changed from white to pastel grey */
        font-size: 16px;
        font-weight: bold;
        padding: 12px 28px;
        border-radius: 10px;
        border: none;
        transition: background 0.3s ease;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #FFABAB;
        transform: scale(1.05);
    }
    .stImage {
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.15);
    }
    .stMarkdown h3 {
        font-size: 22px;
        font-weight: bold;
        color: #FFB5A7;
        margin-top: 25px;
        text-align: center;
    }
    .stMarkdown {
        font-size: 18px;
        color: #FFB5A7;
        text-align: center;
    }
    .card {
        background: #ffffff;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .divider {
        height: 2px;
        background-color: #FFABAB;
        margin: 30px 0;
        border: none;
    }
    .stTextInput label {
        color: grey !important; /* Change the font color to grey */
        font-weight: 600; /* Ensure font weight is applied */
        font-size: 18px; /* Ensure font size is applied */
    }
    </style>
    """, unsafe_allow_html=True)

# Function to reset the page to its initial state
def reset_page():
    # Clear the session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# Streamlit interface
st.title("Micro-Expression Recognition")
st.markdown("Upload onset and apex images to predict the corresponding emotion.")

# Divider
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# File uploader with centered layout and grey font color inside a box
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_onset_image = st.file_uploader("Upload Onset Image", type=["jpg", "jpeg", "png", "bmp"], key="onset_image")
    with col2:
        uploaded_apex_image = st.file_uploader("Upload Apex Image", type=["jpg", "jpeg", "png", "bmp"], key="apex_image")
    st.markdown('</div>', unsafe_allow_html=True)

# Placeholder for images and results
if uploaded_onset_image and uploaded_apex_image:
    onset_img_path = f"temp_onset_{uploaded_onset_image.name}"
    apex_img_path = f"temp_apex_{uploaded_apex_image.name}"

    # Save uploaded images temporarily
    with open(onset_img_path, "wb") as f:
        f.write(uploaded_onset_image.getbuffer())
    with open(apex_img_path, "wb") as f:
        f.write(uploaded_apex_image.getbuffer())

    # Display the uploaded images side by side with a card effect
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(onset_img_path, caption="Onset Image", use_column_width=True)
    with col2:
        st.image(apex_img_path, caption="Apex Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Text input box for manually entering the ground truth emotion with grey font color
    ground_truth_emotion = st.text_input("Enter Ground Truth Emotion:", value="", key="ground_truth")

    # Centralized button to start prediction
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if st.button("Start Predicting"):
        with st.spinner('Predicting...'):
            # Predict emotion
            predicted_emotion, _ = predict_emotion(onset_img_path, apex_img_path)

        # Display results with enhanced styling
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### Results")
        if predicted_emotion:
            st.markdown(f"<h3>Predicted Emotion: {predicted_emotion}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3>Ground Truth Emotion: {ground_truth_emotion}</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>Failed to detect landmarks and predict emotion.</h3>", unsafe_allow_html=True)

        # Add a restart button
        if st.button("Restart"):
            # Delete the temporary image files
            if os.path.exists(onset_img_path):
                os.remove(onset_img_path)
            if os.path.exists(apex_img_path):
                os.remove(apex_img_path)

            reset_page()  # Reset the page to its initial state

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFABAB;'>Please upload both onset and apex images.</h3>", unsafe_allow_html=True)

