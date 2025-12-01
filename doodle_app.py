#!/usr/bin/env python3
"""
Doodle Recognition App - Draw and predict doodles using trained models
Supports multiple model backends: ResNet (classification), ProtoNet (few-shot), and Similarity (OpenCV)
"""

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import random
import pickle
import cv2
from abc import ABC, abstractmethod

# Model paths
MODELS_DIR = "models"

# Game modes
MODE_FREE_DRAW = 0
MODE_CHALLENGE = 1

# Menu overlay colors
OVERLAY_BG = (20, 20, 30, 220)  # Semi-transparent
MENU_BG = (35, 35, 50)
MENU_ITEM_HOVER = (55, 55, 75)
MENU_ITEM_SELECTED = (70, 50, 100)

# Colors - Cyberpunk/Neon aesthetic
BG_COLOR = (15, 15, 25)
CANVAS_BG = (25, 25, 35)
DRAW_COLOR = (255, 255, 255)
ACCENT_1 = (0, 255, 200)  # Cyan
ACCENT_2 = (255, 50, 150)  # Magenta
ACCENT_3 = (150, 100, 255)  # Purple
ACCENT_4 = (255, 200, 50)  # Gold/Yellow for success
TEXT_COLOR = (220, 220, 240)
DIM_TEXT = (100, 100, 120)
BUTTON_COLOR = (40, 40, 60)
BUTTON_HOVER = (60, 60, 90)
PREDICTION_BG = (30, 30, 45)
SUCCESS_COLOR = (50, 255, 100)  # Green for success
MODEL_BADGE_COLOR = (80, 60, 120)  # Purple for model badge

# Window dimensions
WINDOW_WIDTH = 1150
WINDOW_HEIGHT = 750
CANVAS_SIZE = 480
CANVAS_X = 50
CANVAS_Y = 140


# ============================================================================
# Model Backends
# ============================================================================

class ModelBackend(ABC):
    """Abstract base class for model backends"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name"""
        pass
    
    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name for UI badges"""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of categories this model can recognize"""
        pass
    
    @property
    @abstractmethod
    def all_labels(self) -> list:
        """List of all category labels"""
        pass
    
    @abstractmethod
    def predict(self, surface, top_k=10) -> list:
        """
        Predict doodle from pygame surface.
        Returns list of (label, confidence) tuples.
        """
        pass


class ResNetBackend(ModelBackend):
    """ResNet-based classification model"""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        self.model_dir = model_dir
        
        # Load model info
        with open(os.path.join(model_dir, "model_info.json"), 'r') as f:
            info = json.load(f)
        
        self.img_size = info.get('img_size', 160)
        self._num_classes = info.get('num_classes', 340)
        
        # Load label mappings
        with open(os.path.join(model_dir, "label_mappings.json"), 'r') as f:
            mappings = json.load(f)
        self.idx_to_label = {int(k): v for k, v in mappings['idx_to_label'].items()}
        self._all_labels = list(self.idx_to_label.values())
        
        # Create and load model
        self.model = self._create_model()
        model_path = os.path.join(model_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _create_model(self):
        """Create ResNet18 model architecture"""
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self._num_classes)
        )
        return model
    
    @property
    def name(self) -> str:
        return "ResNet-18 Classifier"
    
    @property
    def short_name(self) -> str:
        return "ResNet"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def all_labels(self) -> list:
        return self._all_labels
    
    def predict(self, surface, top_k=10) -> list:
        # Convert pygame surface to PIL Image
        raw_str = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), raw_str)
        
        # Apply transforms
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        predictions = [
            (self.idx_to_label[idx.item()], float(top_probs[0][i].item()))
            for i, idx in enumerate(top_indices[0])
        ]
        
        return predictions


class ConvBlock(nn.Module):
    """Basic convolutional block for ProtoNet"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.block(x)


class ProtoNet(nn.Module):
    """Prototypical Network encoder"""
    def __init__(self, in_channels=1, hidden_dim=64, embedding_dim=64):
        super(ProtoNet, self).__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, embedding_dim),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * 5 * 5, embedding_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class ProtoNetWithDistance(nn.Module):
    """Full Prototypical Network with distance calculation (matches training architecture)"""
    def __init__(self, in_channels=1, hidden_dim=64, embedding_dim=64, distance='euclidean'):
        super(ProtoNetWithDistance, self).__init__()
        self.encoder = ProtoNet(in_channels, hidden_dim, embedding_dim)
        self.distance = distance
    
    def forward(self, x):
        return self.encoder(x)
    
    def get_embedding(self, x):
        return self.encoder(x)


class ProtoNetBackend(ModelBackend):
    """Prototypical Network for few-shot classification"""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        self.model_dir = model_dir
        
        # Load model info
        with open(os.path.join(model_dir, "model_info.json"), 'r') as f:
            info = json.load(f)
        
        self.img_size = info.get('img_size', 84)
        self.embedding_dim = info.get('embedding_dim', 64)
        
        # Load reference embeddings (pre-computed prototypes)
        with open(os.path.join(model_dir, "reference_embeddings.json"), 'r') as f:
            embeddings_dict = json.load(f)
        
        self._all_labels = list(embeddings_dict.keys())
        self._num_classes = len(self._all_labels)
        
        # Convert to tensor
        self.prototypes = torch.tensor(
            [embeddings_dict[label] for label in self._all_labels],
            dtype=torch.float32
        ).to(device)
        
        # Create and load model (uses ProtoNetWithDistance to match training architecture)
        self.model = ProtoNetWithDistance(in_channels=1, hidden_dim=64, embedding_dim=self.embedding_dim)
        model_path = os.path.join(model_dir, "model.pth")
        
        # Load from checkpoint dict (model was saved with optimizer state, etc.)
        # Note: weights_only=False needed for checkpoint dicts with non-tensor data
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Image transforms (grayscale)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    @property
    def name(self) -> str:
        return "Prototypical Network"
    
    @property
    def short_name(self) -> str:
        return "ProtoNet"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def all_labels(self) -> list:
        return self._all_labels
    
    def predict(self, surface, top_k=10) -> list:
        # Convert pygame surface to PIL Image
        raw_str = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), raw_str)
        
        # Apply transforms
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            query_embedding = self.model.get_embedding(img_tensor)
            
            # Compute distances to prototypes (squared Euclidean)
            query_expanded = query_embedding.unsqueeze(1)  # [1, 1, embedding_dim]
            proto_expanded = self.prototypes.unsqueeze(0)  # [1, n_classes, embedding_dim]
            distances = torch.sum((query_expanded - proto_expanded) ** 2, dim=2)  # [1, n_classes]
            
            # Convert to probabilities (negative distance -> softmax)
            probs = F.softmax(-distances, dim=1).squeeze()
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self._all_labels)))
        predictions = [
            (self._all_labels[idx.item()], float(top_probs[i].item()))
            for i, idx in enumerate(top_indices)
        ]
        
        return predictions


class CV2SimilarityClassifier:
    """
    Classifier using OpenCV's template matching and feature matching.
    Uses multiple similarity metrics from cv2.
    This class needs to be defined here so pickle can load saved models.
    """
    
    def __init__(self, method='multi'):
        """
        Args:
            method: 'template' (cv2.matchTemplate), 'features' (keypoint matching), 
                   'histogram' (histogram comparison), or 'multi' (combines all)
        """
        self.method = method
        self.templates = {}
        self.classes_ = None
        
        # Initialize feature detectors
        if method in ['features', 'multi']:
            try:
                # Try SIFT first (better but requires opencv-contrib-python)
                self.detector = cv2.SIFT_create()
                self.matcher = cv2.BFMatcher()
            except:
                # Fall back to ORB (built-in)
                self.detector = cv2.ORB_create()
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def fit(self, X, y):
        """Create template prototypes from training images."""
        self.classes_ = np.unique(y)
        
        for cls in self.classes_:
            class_samples = X[y == cls]
            # Store multiple templates per class (top samples)
            # Use mean as primary template
            self.templates[cls] = {
                'mean': np.mean(class_samples, axis=0).astype(np.uint8),
                'samples': class_samples[:10].astype(np.uint8)  # Store top 10 samples
            }
            
            # Precompute keypoints and descriptors for feature matching
            if self.method in ['features', 'multi']:
                self.templates[cls]['keypoints'] = []
                self.templates[cls]['descriptors'] = []
                for sample in self.templates[cls]['samples']:
                    kp, desc = self.detector.detectAndCompute(sample, None)
                    self.templates[cls]['keypoints'].append(kp)
                    self.templates[cls]['descriptors'].append(desc)
        
        return self
    
    def _template_match_score(self, img, template):
        """Use cv2.matchTemplate with multiple methods."""
        # Normalize images
        img_norm = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        template_norm = cv2.normalize(template.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        # Try different matching methods
        methods = [
            cv2.TM_CCOEFF_NORMED,  # Normalized correlation coefficient
            cv2.TM_CCORR_NORMED,   # Normalized cross-correlation
        ]
        
        scores = []
        for method in methods:
            result = cv2.matchTemplate(img_norm, template_norm, method)
            scores.append(np.max(result))
        
        return np.mean(scores)
    
    def _feature_match_score(self, img, template_data):
        """Match using keypoint features."""
        kp_img, desc_img = self.detector.detectAndCompute(img, None)
        
        if desc_img is None or len(kp_img) < 4:
            return 0.0
        
        best_matches = []
        for desc_template in template_data['descriptors']:
            if desc_template is None:
                continue
            
            try:
                # Check if matcher has getCrossCheck method (ORB)
                if hasattr(self.matcher, 'getCrossCheck') and self.matcher.getCrossCheck():
                    # ORB with Hamming distance
                    matches = self.matcher.match(desc_img, desc_template)
                    matches = sorted(matches, key=lambda x: x.distance)
                    best_matches.extend(matches[:20])  # Top 20 matches
                else:
                    # SIFT with ratio test
                    matches = self.matcher.knnMatch(desc_img, desc_template, k=2)
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                                good_matches.append(m)
                    best_matches.extend(good_matches)
            except:
                continue
        
        if len(best_matches) == 0:
            return 0.0
        
        # Score based on number of good matches
        match_score = len(best_matches) / max(len(kp_img), 1)
        return min(match_score, 1.0)
    
    def _histogram_match_score(self, img, template):
        """Compare histograms using multiple methods."""
        # Calculate histograms
        hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_template = cv2.calcHist([template], [0], None, [256], [0, 256])
        
        # Normalize
        cv2.normalize(hist_img, hist_img, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
        
        # Compare using multiple methods
        methods = [
            cv2.HISTCMP_CORREL,      # Correlation
            cv2.HISTCMP_INTERSECT,   # Intersection
            cv2.HISTCMP_BHATTACHARYYA # Bhattacharyya distance
        ]
        
        scores = []
        for method in methods:
            score = cv2.compareHist(hist_img, hist_template, method)
            if method == cv2.HISTCMP_BHATTACHARYYA:
                # Lower is better for Bhattacharyya, convert to similarity
                score = 1.0 - min(score, 1.0)
            scores.append(score)
        
        return np.mean(scores)
    
    def _compute_similarity(self, img, template_data):
        """Compute similarity using selected method(s)."""
        template = template_data['mean']
        scores = []
        
        if self.method in ['template', 'multi']:
            scores.append(self._template_match_score(img, template))
        
        if self.method in ['features', 'multi']:
            scores.append(self._feature_match_score(img, template_data))
        
        if self.method in ['histogram', 'multi']:
            scores.append(self._histogram_match_score(img, template))
        
        return np.mean(scores) if scores else 0.0


class SimilarityBackend(ModelBackend):
    """OpenCV-based similarity matching model"""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.device = device  # Not used but kept for interface consistency
        self.model_dir = model_dir
        
        # Load model info if it exists, otherwise use defaults
        model_info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                info = json.load(f)
            self.img_size = info.get('img_size', 64)
            self.method = info.get('method', 'multi')
        else:
            self.img_size = 64
            self.method = 'multi'
        
        # Load label mappings
        with open(os.path.join(model_dir, "label_mappings.json"), 'r') as f:
            mappings = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in mappings['idx_to_label'].items()}
        self._all_labels = list(self.idx_to_label.values())
        self._num_classes = len(self._all_labels)
        
        # Load the classifier from pickle
        classifier_path = os.path.join(model_dir, "classifier.pkl")
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Ensure method matches (use classifier's method if available, otherwise use from model_info)
        if hasattr(self.classifier, 'method'):
            self.method = self.classifier.method
        
        # Recreate detector and matcher if needed (they might not be pickleable)
        if self.method in ['features', 'multi']:
            if not hasattr(self.classifier, 'detector') or self.classifier.detector is None:
                try:
                    self.classifier.detector = cv2.SIFT_create()
                    self.classifier.matcher = cv2.BFMatcher()
                except:
                    self.classifier.detector = cv2.ORB_create()
                    self.classifier.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    @property
    def name(self) -> str:
        method_name = self.method.capitalize() if self.method != 'multi' else 'Multi-Method'
        return f"OpenCV Similarity ({method_name})"
    
    @property
    def short_name(self) -> str:
        return "Similarity"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def all_labels(self) -> list:
        return self._all_labels
    
    def predict(self, surface, top_k=10) -> list:
        # Convert pygame surface to numpy array (grayscale)
        raw_str = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), raw_str)
        
        # Convert to grayscale numpy array
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Resize to model's expected size
        img_resized = cv2.resize(img_array, (self.img_size, self.img_size))
        
        # Invert if needed (doodles are typically white on black)
        if img_resized.mean() > 127:
            img_resized = 255 - img_resized
        
        # Convert to uint8
        img_uint8 = img_resized.astype(np.uint8)
        
        # Compute similarities for all classes
        similarities = {}
        for cls_idx in self.classifier.classes_:
            template_data = self.classifier.templates[cls_idx]
            similarity = self.classifier._compute_similarity(img_uint8, template_data)
            label = self.idx_to_label[cls_idx]
            similarities[label] = similarity
        
        # Sort by similarity and get top k
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize similarities to probabilities using softmax
        # This ensures all probabilities sum to 1 and are in [0, 1] range
        sim_values = [sim for _, sim in sorted_similarities[:top_k]]
        
        if not sim_values:
            return []
        
        # Use softmax normalization for proper probability distribution
        # Shift by max for numerical stability
        max_sim = max(sim_values)
        exp_sims = [np.exp((sim - max_sim) * 5) for sim in sim_values]  # Scale factor for sharper distribution
        sum_exp = sum(exp_sims)
        
        predictions = []
        for i, (label, similarity) in enumerate(sorted_similarities[:top_k]):
            # Convert to probability using softmax
            prob = exp_sims[i] / sum_exp if sum_exp > 0 else 1.0 / len(sim_values)
            predictions.append((label, float(prob)))
        
        return predictions


def discover_models() -> dict:
    """Discover available models in the models directory and similarity_model directory"""
    models_found = {}
    
    # Check models directory
    if os.path.exists(MODELS_DIR):
        for model_name in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_name)
            if not os.path.isdir(model_path):
                continue
            
            # Check for PyTorch models (model.pth)
            if os.path.exists(os.path.join(model_path, "model.pth")):
                models_found[model_name] = model_path
            # Check for similarity models (classifier.pkl)
            elif os.path.exists(os.path.join(model_path, "classifier.pkl")):
                models_found[model_name] = model_path
    
    # Also check similarity_model directory (legacy location)
    similarity_model_dir = "similarity_model"
    if os.path.exists(similarity_model_dir) and os.path.exists(os.path.join(similarity_model_dir, "classifier.pkl")):
        models_found["similarity"] = similarity_model_dir
    
    return models_found


def load_model_backend(model_type: str, model_dir: str, device: torch.device) -> ModelBackend:
    """Load appropriate model backend based on type"""
    if model_type == "resnet":
        return ResNetBackend(model_dir, device)
    elif model_type == "protonet":
        return ProtoNetBackend(model_dir, device)
    elif model_type == "similarity":
        return SimilarityBackend(model_dir, device)
    else:
        # Try to auto-detect based on files present
        if os.path.exists(os.path.join(model_dir, "classifier.pkl")):
            return SimilarityBackend(model_dir, device)
        elif os.path.exists(os.path.join(model_dir, "model.pth")):
            # Check if it's resnet or protonet based on model_info.json
            if os.path.exists(os.path.join(model_dir, "model_info.json")):
                with open(os.path.join(model_dir, "model_info.json"), 'r') as f:
                    info = json.load(f)
                if 'embedding_dim' in info:
                    return ProtoNetBackend(model_dir, device)
                else:
                    return ResNetBackend(model_dir, device)
        raise ValueError(f"Unknown model type: {model_type} and could not auto-detect")


# ============================================================================
# UI Components
# ============================================================================

def draw_rounded_rect(surface, rect, color, radius=10):
    """Draw a rounded rectangle"""
    x, y, w, h = rect
    pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
    pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))
    pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + radius, y + h - radius), radius)
    pygame.draw.circle(surface, color, (x + w - radius, y + h - radius), radius)


def draw_gradient_bar(surface, x, y, width, height, progress, color_start, color_end):
    """Draw a gradient progress bar"""
    if progress <= 0:
        return
    bar_width = int(width * min(progress, 1.0))
    for i in range(bar_width):
        t = i / max(width - 1, 1)
        r = int(color_start[0] + (color_end[0] - color_start[0]) * t)
        g = int(color_start[1] + (color_end[1] - color_start[1]) * t)
        b = int(color_start[2] + (color_end[2] - color_start[2]) * t)
        pygame.draw.line(surface, (r, g, b), (x + i, y), (x + i, y + height - 1))


class Button:
    def __init__(self, x, y, width, height, text, action, color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        self.color = color  # Optional custom color
        
    def draw(self, surface, font):
        if self.color:
            color = self.color if not self.hovered else tuple(min(c + 20, 255) for c in self.color)
        else:
            color = BUTTON_HOVER if self.hovered else BUTTON_COLOR
        draw_rounded_rect(surface, self.rect, color, 8)
        
        # Border
        border_color = ACCENT_1 if self.hovered else DIM_TEXT
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=8)
        
        # Text
        text_surface = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.action()
                return True
        return False


# ============================================================================
# Main Application
# ============================================================================

class DoodleApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("✨ Doodle Recognition")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create canvas
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas.fill(CANVAS_BG)
        
        # Drawing state
        self.drawing = False
        self.last_pos = None
        self.brush_size = 8
        self.has_drawn = False
        self.mouse_pos = (0, 0)
        
        # Undo/Redo history
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50
        
        # Predictions
        self.predictions = []
        self.prediction_timer = 0
        self.prediction_interval = 500
        
        # Game mode
        self.mode = MODE_FREE_DRAW
        self.target_word = None
        self.challenge_success = False
        self.success_time = 0
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.success_rank = 0
        
        # Model menu state
        self.show_model_menu = False
        self.menu_hovered_index = -1
        
        # Setup device
        self.device = self._setup_device()
        
        # Discover and load models
        self.available_models = discover_models()
        self.model_backends = {}
        self.current_model_key = None
        self.current_model = None
        
        self._load_all_models()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 56)
        self.challenge_font = pygame.font.Font(None, 60)
        self.heading_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 30)
        self.body_font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        
        # Buttons
        self.update_buttons()
        
    def _setup_device(self):
        """Setup compute device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    def _load_all_models(self):
        """Load all available models"""
        print("Discovering models...")
        print(f"  Found {len(self.available_models)} model(s): {list(self.available_models.keys())}")
        
        for model_key, model_path in self.available_models.items():
            try:
                print(f"  Loading {model_key}...")
                backend = load_model_backend(model_key, model_path, self.device)
                self.model_backends[model_key] = backend
                print(f"    ✓ {backend.name} ({backend.num_classes} categories)")
                
                if self.current_model_key is None:
                    self.current_model_key = model_key
                    self.current_model = backend
            except Exception as e:
                print(f"    ✗ Failed to load {model_key}: {e}")
        
        if not self.model_backends:
            raise RuntimeError("No models found in models/ directory!")
        
        print(f"\nActive model: {self.current_model.name}")
    
    def switch_model(self, model_key: str):
        """Switch to a different model"""
        if model_key in self.model_backends and model_key != self.current_model_key:
            self.current_model_key = model_key
            self.current_model = self.model_backends[model_key]
            self.predictions = []  # Clear predictions
            print(f"Switched to: {self.current_model.name}")
            
            # If in challenge mode, pick a new word from new model's vocabulary
            if self.mode == MODE_CHALLENGE:
                self.next_challenge()
    
    def cycle_model(self):
        """Cycle to the next available model"""
        keys = list(self.model_backends.keys())
        if len(keys) <= 1:
            return
        
        current_idx = keys.index(self.current_model_key)
        next_idx = (current_idx + 1) % len(keys)
        self.switch_model(keys[next_idx])
        self.update_buttons()
    
    def toggle_model_menu(self):
        """Toggle the model selection menu"""
        self.show_model_menu = not self.show_model_menu
        self.menu_hovered_index = -1
    
    def get_model_menu_rects(self):
        """Get the rects for model menu items"""
        menu_width = 350
        menu_item_height = 60
        menu_padding = 15
        num_models = len(self.model_backends)
        menu_height = num_models * menu_item_height + menu_padding * 2 + 50  # +50 for title
        
        menu_x = (WINDOW_WIDTH - menu_width) // 2
        menu_y = (WINDOW_HEIGHT - menu_height) // 2
        
        rects = []
        keys = list(self.model_backends.keys())
        for i, key in enumerate(keys):
            item_y = menu_y + 50 + menu_padding + i * menu_item_height
            rect = pygame.Rect(menu_x + menu_padding, item_y, 
                              menu_width - menu_padding * 2, menu_item_height - 8)
            rects.append((key, rect))
        
        return menu_x, menu_y, menu_width, menu_height, rects
    
    def draw_model_menu(self):
        """Draw the model selection menu overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((20, 20, 30, 200))
        self.screen.blit(overlay, (0, 0))
        
        menu_x, menu_y, menu_width, menu_height, item_rects = self.get_model_menu_rects()
        
        # Menu background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        draw_rounded_rect(self.screen, menu_rect, MENU_BG, 15)
        pygame.draw.rect(self.screen, ACCENT_3, menu_rect, 2, border_radius=15)
        
        # Title
        title = self.heading_font.render("SELECT MODEL", True, ACCENT_1)
        title_rect = title.get_rect(centerx=menu_x + menu_width // 2, top=menu_y + 15)
        self.screen.blit(title, title_rect)
        
        # Model items
        keys = list(self.model_backends.keys())
        for i, (key, rect) in enumerate(item_rects):
            backend = self.model_backends[key]
            is_selected = key == self.current_model_key
            is_hovered = i == self.menu_hovered_index
            
            # Item background
            if is_selected:
                item_color = MENU_ITEM_SELECTED
            elif is_hovered:
                item_color = MENU_ITEM_HOVER
            else:
                item_color = MENU_BG
            
            draw_rounded_rect(self.screen, rect, item_color, 8)
            
            # Border for selected/hovered
            if is_selected:
                pygame.draw.rect(self.screen, ACCENT_1, rect, 2, border_radius=8)
            elif is_hovered:
                pygame.draw.rect(self.screen, DIM_TEXT, rect, 1, border_radius=8)
            
            # Model name
            name_color = ACCENT_1 if is_selected else TEXT_COLOR
            name_text = self.label_font.render(backend.name, True, name_color)
            self.screen.blit(name_text, (rect.x + 15, rect.y + 8))
            
            # Model info
            info_text = f"{backend.num_classes} categories"
            info_surface = self.small_font.render(info_text, True, DIM_TEXT)
            self.screen.blit(info_surface, (rect.x + 15, rect.y + 32))
            
            # Checkmark for selected
            if is_selected:
                check = self.label_font.render("✓", True, SUCCESS_COLOR)
                self.screen.blit(check, (rect.right - 35, rect.y + 15))
        
        # Instructions
        hint = self.small_font.render("Click to select  •  M or ESC to close", True, DIM_TEXT)
        hint_rect = hint.get_rect(centerx=menu_x + menu_width // 2, 
                                  bottom=menu_y + menu_height - 10)
        self.screen.blit(hint, hint_rect)
    
    def handle_model_menu_event(self, event):
        """Handle events for the model menu"""
        if not self.show_model_menu:
            return False
        
        menu_x, menu_y, menu_width, menu_height, item_rects = self.get_model_menu_rects()
        
        if event.type == pygame.MOUSEMOTION:
            self.menu_hovered_index = -1
            for i, (key, rect) in enumerate(item_rects):
                if rect.collidepoint(event.pos):
                    self.menu_hovered_index = i
                    break
            return True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check if clicked on a menu item
                for i, (key, rect) in enumerate(item_rects):
                    if rect.collidepoint(event.pos):
                        self.switch_model(key)
                        self.show_model_menu = False
                        self.update_buttons()
                        return True
                
                # Check if clicked outside menu to close
                menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                if not menu_rect.collidepoint(event.pos):
                    self.show_model_menu = False
                    return True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_m:
                self.show_model_menu = False
                return True
            elif event.key == pygame.K_UP:
                keys = list(self.model_backends.keys())
                if self.menu_hovered_index <= 0:
                    self.menu_hovered_index = len(keys) - 1
                else:
                    self.menu_hovered_index -= 1
                return True
            elif event.key == pygame.K_DOWN:
                keys = list(self.model_backends.keys())
                self.menu_hovered_index = (self.menu_hovered_index + 1) % len(keys)
                return True
            elif event.key == pygame.K_RETURN:
                keys = list(self.model_backends.keys())
                if 0 <= self.menu_hovered_index < len(keys):
                    self.switch_model(keys[self.menu_hovered_index])
                    self.show_model_menu = False
                    self.update_buttons()
                return True
        
        return False
    
    def update_buttons(self):
        """Update buttons based on current mode"""
        btn_w = 75
        gap = 6
        btn_h = 40
        x = CANVAS_X
        y = CANVAS_Y + CANVAS_SIZE + 15
        
        if self.mode == MODE_FREE_DRAW:
            self.buttons = [
                Button(x, y, btn_w, btn_h, "Undo", self.undo),
                Button(x + (btn_w + gap), y, btn_w, btn_h, "Redo", self.redo),
                Button(x + 2*(btn_w + gap), y, btn_w, btn_h, "Clear", self.clear_canvas),
                Button(x + 3*(btn_w + gap), y, btn_w, btn_h, "−", self.decrease_brush),
                Button(x + 4*(btn_w + gap), y, btn_w, btn_h, "+", self.increase_brush),
                Button(x + 5*(btn_w + gap), y, 100, btn_h, "Challenge ▶", self.switch_to_challenge),
            ]
        else:
            skip_label = "Next ▶" if self.challenge_success else "Skip"
            skip_action = self.next_challenge if self.challenge_success else self.skip_challenge
            self.buttons = [
                Button(x, y, btn_w, btn_h, "Undo", self.undo),
                Button(x + (btn_w + gap), y, btn_w, btn_h, "Redo", self.redo),
                Button(x + 2*(btn_w + gap), y, btn_w, btn_h, "Clear", self.clear_canvas),
                Button(x + 3*(btn_w + gap), y, btn_w, btn_h, "−", self.decrease_brush),
                Button(x + 4*(btn_w + gap), y, btn_w, btn_h, "+", self.increase_brush),
                Button(x + 5*(btn_w + gap), y, 100, btn_h, skip_label, skip_action),
            ]
    
    def skip_challenge(self):
        """Skip current challenge (resets streak)"""
        self.streak = 0
        self.next_challenge()
    
    def switch_to_challenge(self):
        """Switch to challenge mode"""
        self.mode = MODE_CHALLENGE
        self.score = 0
        self.streak = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.next_challenge()
        self.update_buttons()
        
    def switch_to_free_draw(self):
        """Switch to free draw mode"""
        self.mode = MODE_FREE_DRAW
        self.target_word = None
        self.challenge_success = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_canvas(save_history=False)
        self.update_buttons()
        
    def next_challenge(self):
        """Pick a new random word to draw"""
        self.target_word = random.choice(self.current_model.all_labels)
        self.challenge_success = False
        self.success_rank = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_canvas(save_history=False)
        self.update_buttons()
    
    def save_canvas_state(self):
        """Save current canvas state for undo"""
        state = self.canvas.copy()
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        
    def undo(self):
        """Undo last stroke"""
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
            self.has_drawn = len(self.undo_stack) > 0
            self.predictions = []
            
    def redo(self):
        """Redo last undone stroke"""
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
            self.has_drawn = True
            self.predictions = []
    
    def clear_canvas(self, save_history=True):
        """Clear the drawing canvas"""
        if self.has_drawn and save_history:
            self.save_canvas_state()
        self.canvas.fill(CANVAS_BG)
        self.predictions = []
        self.has_drawn = False
        
    def increase_brush(self):
        """Increase brush size"""
        self.brush_size = min(25, self.brush_size + 2)
        
    def decrease_brush(self):
        """Decrease brush size"""
        self.brush_size = max(2, self.brush_size - 2)
        
    def draw_on_canvas(self, pos):
        """Draw on canvas at position"""
        canvas_x = pos[0] - CANVAS_X
        canvas_y = pos[1] - CANVAS_Y
        
        if 0 <= canvas_x < CANVAS_SIZE and 0 <= canvas_y < CANVAS_SIZE:
            if self.last_pos:
                last_x = self.last_pos[0] - CANVAS_X
                last_y = self.last_pos[1] - CANVAS_Y
                pygame.draw.line(self.canvas, DRAW_COLOR, 
                               (last_x, last_y), (canvas_x, canvas_y), 
                               self.brush_size)
            pygame.draw.circle(self.canvas, DRAW_COLOR, (canvas_x, canvas_y), self.brush_size // 2)
            self.last_pos = pos
            self.has_drawn = True
            
    def update_predictions(self):
        """Update predictions from model"""
        if self.has_drawn:
            self.predictions = self.current_model.predict(self.canvas)
            
            # Check for challenge success
            if self.mode == MODE_CHALLENGE and self.target_word:
                for i, (label, confidence) in enumerate(self.predictions[:5]):
                    if label.lower() == self.target_word.lower():
                        new_rank = i + 1
                        if not self.challenge_success:
                            self.challenge_success = True
                            self.success_time = pygame.time.get_ticks()
                            self.success_rank = new_rank
                            self.streak += 1
                            self.best_streak = max(self.best_streak, self.streak)
                            points = [100, 50, 30, 20, 10][i]
                            self.score += points
                            self.update_buttons()
                        elif new_rank < self.success_rank:
                            old_points = [100, 50, 30, 20, 10][self.success_rank - 1]
                            new_points = [100, 50, 30, 20, 10][i]
                            bonus = new_points - old_points
                            self.score += bonus
                            self.success_rank = new_rank
                        break
            
    def draw_header(self):
        """Draw the header section"""
        if self.mode == MODE_FREE_DRAW:
            self._draw_free_draw_header()
        else:
            self._draw_challenge_header()
    
    def _draw_model_badge(self, x, y):
        """Draw current model badge"""
        badge_text = self.current_model.short_name
        badge_surface = self.small_font.render(badge_text, True, ACCENT_3)
        badge_width = badge_surface.get_width() + 16
        badge_height = 24
        
        # Background
        draw_rounded_rect(self.screen, (x, y, badge_width, badge_height), MODEL_BADGE_COLOR, 5)
        pygame.draw.rect(self.screen, ACCENT_3, (x, y, badge_width, badge_height), 1, border_radius=5)
        
        # Text
        self.screen.blit(badge_surface, (x + 8, y + 4))
        
        return badge_width
            
    def _draw_free_draw_header(self):
        """Draw header for free draw mode"""
        title_text = "DOODLE RECOGNITION"
        
        # Glow
        glow_surface = self.title_font.render(title_text, True, ACCENT_1)
        glow_rect = glow_surface.get_rect(centerx=WINDOW_WIDTH//2, top=20)
        
        for offset in range(3, 0, -1):
            glow = self.title_font.render(title_text, True, ACCENT_1)
            self.screen.blit(glow, (glow_rect.x - offset, glow_rect.y))
            self.screen.blit(glow, (glow_rect.x + offset, glow_rect.y))
        
        self.screen.blit(glow_surface, glow_rect)
        
        # Model badge and subtitle
        badge_y = 70
        model_text = f"Model: "
        model_surface = self.body_font.render(model_text, True, DIM_TEXT)
        model_width = model_surface.get_width()
        
        total_width = model_width + 100  # approximate badge width
        start_x = WINDOW_WIDTH//2 - total_width//2
        
        self.screen.blit(model_surface, (start_x, badge_y))
        badge_w = self._draw_model_badge(start_x + model_width + 4, badge_y - 2)
        
        # Category count
        cat_text = f" · {self.current_model.num_classes} categories"
        cat_surface = self.body_font.render(cat_text, True, DIM_TEXT)
        self.screen.blit(cat_surface, (start_x + model_width + badge_w + 12, badge_y))
        
        # Mode hints
        mode_hint = self.small_font.render("TAB Challenge  •  M Switch Model", True, DIM_TEXT)
        self.screen.blit(mode_hint, (WINDOW_WIDTH - 260, 25))
        
    def _draw_challenge_header(self):
        """Draw header for challenge mode"""
        # Left side: Score and streak
        mode_text = self.body_font.render("CHALLENGE MODE", True, ACCENT_2)
        self.screen.blit(mode_text, (CANVAS_X, 15))
        
        score_text = self.heading_font.render(f"Score: {self.score}", True, ACCENT_4)
        self.screen.blit(score_text, (CANVAS_X, 42))
        
        streak_color = SUCCESS_COLOR if self.streak >= 3 else TEXT_COLOR
        streak_text = self.body_font.render(f"Streak: {self.streak}  Best: {self.best_streak}", True, streak_color)
        self.screen.blit(streak_text, (CANVAS_X, 75))
        
        # Model badge
        self._draw_model_badge(CANVAS_X, 98)
        
        # Right side: Back hint
        back_text = self.small_font.render("ESC to exit  •  M model", True, DIM_TEXT)
        self.screen.blit(back_text, (WINDOW_WIDTH - 160, 15))
        
        # Center: Target word
        if self.target_word:
            word_color = SUCCESS_COLOR if self.challenge_success else ACCENT_1
            word_surface = self.challenge_font.render(self.target_word.upper(), True, word_color)
            word_rect = word_surface.get_rect(centerx=WINDOW_WIDTH//2, top=15)
            
            for offset in range(2, 0, -1):
                glow = self.challenge_font.render(self.target_word.upper(), True, word_color)
                self.screen.blit(glow, (word_rect.x - offset, word_rect.y))
                self.screen.blit(glow, (word_rect.x + offset, word_rect.y))
            
            self.screen.blit(word_surface, word_rect)
            
            prompt = self.small_font.render("DRAW THIS:", True, DIM_TEXT)
            prompt_rect = prompt.get_rect(centerx=WINDOW_WIDTH//2, bottom=word_rect.top - 2)
            self.screen.blit(prompt, prompt_rect)
            
            if self.challenge_success:
                if self.success_rank == 1:
                    success_msg = "🏆 PERFECT! #1"
                else:
                    success_msg = f"✓ Found at #{self.success_rank} - Keep drawing for #1!"
                success_surface = self.body_font.render(success_msg, True, SUCCESS_COLOR)
                success_rect = success_surface.get_rect(centerx=WINDOW_WIDTH//2, top=75)
                self.screen.blit(success_surface, success_rect)
                
                next_hint = self.small_font.render("SPACE for next challenge", True, ACCENT_4)
                next_rect = next_hint.get_rect(centerx=WINDOW_WIDTH//2, top=100)
                self.screen.blit(next_hint, next_rect)
        
    def draw_canvas_area(self):
        """Draw the canvas area with border"""
        border_rect = pygame.Rect(CANVAS_X - 3, CANVAS_Y - 3, CANVAS_SIZE + 6, CANVAS_SIZE + 6)
        pygame.draw.rect(self.screen, ACCENT_1, border_rect, 2, border_radius=5)
        
        label = self.heading_font.render("CANVAS", True, TEXT_COLOR)
        self.screen.blit(label, (CANVAS_X, CANVAS_Y - 38))
        
        # Brush size indicator
        brush_label = self.small_font.render("BRUSH", True, DIM_TEXT)
        brush_label_x = CANVAS_X + CANVAS_SIZE - 90
        self.screen.blit(brush_label, (brush_label_x, CANVAS_Y - 38))
        
        preview_x = CANVAS_X + CANVAS_SIZE - 25
        preview_y = CANVAS_Y - 22
        pygame.draw.circle(self.screen, BUTTON_COLOR, (preview_x, preview_y), 18)
        pygame.draw.circle(self.screen, DIM_TEXT, (preview_x, preview_y), 18, 1)
        preview_radius = min(self.brush_size // 2, 14)
        pygame.draw.circle(self.screen, DRAW_COLOR, (preview_x, preview_y), preview_radius)
        
        size_text = self.small_font.render(f"{self.brush_size}px", True, ACCENT_1)
        size_rect = size_text.get_rect(centerx=preview_x, top=preview_y + 22)
        self.screen.blit(size_text, size_rect)
        
        self.screen.blit(self.canvas, (CANVAS_X, CANVAS_Y))
        self.draw_brush_cursor()
    
    def draw_brush_cursor(self):
        """Draw brush preview circle at mouse position when over canvas"""
        mx, my = self.mouse_pos
        if (CANVAS_X <= mx < CANVAS_X + CANVAS_SIZE and 
            CANVAS_Y <= my < CANVAS_Y + CANVAS_SIZE):
            radius = self.brush_size // 2
            pygame.draw.circle(self.screen, ACCENT_1, (mx, my), radius + 2, 1)
            pygame.draw.circle(self.screen, DRAW_COLOR, (mx, my), radius, 1)
            if radius > 3:
                pygame.draw.circle(self.screen, ACCENT_1, (mx, my), 1)
        
    def draw_predictions(self):
        """Draw the predictions panel"""
        panel_x = CANVAS_X + CANVAS_SIZE + 30
        panel_y = CANVAS_Y
        panel_width = WINDOW_WIDTH - panel_x - 40
        panel_height = CANVAS_SIZE
        
        draw_rounded_rect(self.screen, (panel_x, panel_y, panel_width, panel_height), PREDICTION_BG, 12)
        pygame.draw.rect(self.screen, ACCENT_2, (panel_x, panel_y, panel_width, panel_height), 2, border_radius=12)
        
        title = self.heading_font.render("PREDICTIONS", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 15, panel_y + 12))
        
        if not self.predictions:
            empty_text = self.body_font.render("Start drawing to see predictions...", True, DIM_TEXT)
            empty_rect = empty_text.get_rect(center=(panel_x + panel_width//2, panel_y + panel_height//2))
            self.screen.blit(empty_text, empty_rect)
        else:
            start_y = panel_y + 50
            bar_width = panel_width - 40
            row_height = 46
            
            for i, (label, confidence) in enumerate(self.predictions):
                y = start_y + i * row_height
                
                is_target = (self.mode == MODE_CHALLENGE and 
                           self.target_word and 
                           label.lower() == self.target_word.lower())
                
                if is_target:
                    highlight_rect = pygame.Rect(panel_x + 8, y - 4, panel_width - 16, row_height - 4)
                    pygame.draw.rect(self.screen, (40, 80, 50), highlight_rect, border_radius=5)
                    pygame.draw.rect(self.screen, SUCCESS_COLOR, highlight_rect, 2, border_radius=5)
                
                rank_colors = [ACCENT_1, ACCENT_2, ACCENT_3]
                if is_target:
                    rank_color = SUCCESS_COLOR
                else:
                    rank_color = rank_colors[i] if i < 3 else DIM_TEXT
                rank_text = self.body_font.render(f"#{i+1}", True, rank_color)
                self.screen.blit(rank_text, (panel_x + 15, y))
                
                label_display = label[:18] + "..." if len(label) > 18 else label
                label_color = SUCCESS_COLOR if is_target else TEXT_COLOR
                label_text = self.body_font.render(label_display, True, label_color)
                self.screen.blit(label_text, (panel_x + 55, y))
                
                if is_target:
                    conf_color = SUCCESS_COLOR
                else:
                    conf_color = ACCENT_1 if confidence > 0.5 else TEXT_COLOR
                conf_text = self.body_font.render(f"{confidence*100:.1f}%", True, conf_color)
                self.screen.blit(conf_text, (panel_x + panel_width - 65, y))
                
                bar_y = y + 24
                bar_height = 5
                
                pygame.draw.rect(self.screen, BUTTON_COLOR, 
                               (panel_x + 55, bar_y, bar_width - 80, bar_height), border_radius=3)
                
                if confidence > 0.01:
                    if is_target:
                        draw_gradient_bar(self.screen, panel_x + 55, bar_y, 
                                        bar_width - 80, bar_height, confidence,
                                        (30, 150, 60), SUCCESS_COLOR)
                    else:
                        draw_gradient_bar(self.screen, panel_x + 55, bar_y, 
                                        bar_width - 80, bar_height, confidence,
                                        ACCENT_2, ACCENT_1)
                    
    def draw_instructions(self):
        """Draw keyboard instructions"""
        panel_x = CANVAS_X + CANVAS_SIZE + 30
        y = CANVAS_Y + CANVAS_SIZE + 18
        
        shortcuts_label = self.small_font.render("SHORTCUTS", True, DIM_TEXT)
        self.screen.blit(shortcuts_label, (panel_x, y))
        
        if self.mode == MODE_FREE_DRAW:
            shortcuts = "⌘Z Undo  •  ⌘Y Redo  •  C Clear  •  M Model  •  TAB Challenge"
        else:
            shortcuts = "⌘Z Undo  •  SPACE Next  •  C Clear  •  M Model  •  ESC Back"
        
        shortcuts_text = self.small_font.render(shortcuts, True, ACCENT_1)
        self.screen.blit(shortcuts_text, (panel_x + 85, y))
        
        history_text = f"Undo: {len(self.undo_stack)} | Redo: {len(self.redo_stack)}"
        history_surface = self.small_font.render(history_text, True, DIM_TEXT)
        history_rect = history_surface.get_rect(right=WINDOW_WIDTH - 40, top=y)
        self.screen.blit(history_surface, history_rect)
            
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            
            # Handle model menu events first if it's open
            if self.show_model_menu:
                if self.handle_model_menu_event(event):
                    continue
                
            if event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                ctrl = mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META
                shift = mods & pygame.KMOD_SHIFT
                
                if event.key == pygame.K_ESCAPE:
                    if self.show_model_menu:
                        self.show_model_menu = False
                    elif self.mode == MODE_CHALLENGE:
                        self.switch_to_free_draw()
                    else:
                        self.running = False
                elif event.key == pygame.K_z and ctrl:
                    if shift:
                        self.redo()
                    else:
                        self.undo()
                elif event.key == pygame.K_y and ctrl:
                    self.redo()
                elif event.key == pygame.K_c and not ctrl:
                    self.clear_canvas()
                elif event.key == pygame.K_m and not ctrl:
                    self.toggle_model_menu()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.increase_brush()
                elif event.key == pygame.K_MINUS:
                    self.decrease_brush()
                elif event.key == pygame.K_SPACE:
                    if self.mode == MODE_CHALLENGE:
                        if self.challenge_success:
                            self.next_challenge()
                        else:
                            self.next_challenge()
                            self.streak = 0
                elif event.key == pygame.K_TAB:
                    if self.mode == MODE_FREE_DRAW:
                        self.switch_to_challenge()
                    else:
                        self.switch_to_free_draw()
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not self.show_model_menu:
                    self.save_canvas_state()
                    self.drawing = True
                    self.draw_on_canvas(event.pos)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
                    self.last_pos = None
                    
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                if self.drawing and not self.show_model_menu:
                    self.draw_on_canvas(event.pos)
            
            # Don't process button events if menu is open
            if not self.show_model_menu:
                for button in self.buttons:
                    button.handle_event(event)
                
    def run(self):
        """Main game loop"""
        last_prediction_time = 0
        last_success_state = False
        
        while self.running:
            current_time = pygame.time.get_ticks()
            
            self.handle_events()
            
            if self.has_drawn and current_time - last_prediction_time > self.prediction_interval:
                self.update_predictions()
                last_prediction_time = current_time
            
            if self.mode == MODE_CHALLENGE and self.challenge_success != last_success_state:
                self.update_buttons()
                last_success_state = self.challenge_success
            
            self.screen.fill(BG_COLOR)
            pygame.draw.line(self.screen, (35, 35, 50), (0, 120), (WINDOW_WIDTH, 120), 1)
            
            self.draw_header()
            self.draw_canvas_area()
            self.draw_predictions()
            
            for button in self.buttons:
                button.draw(self.screen, self.body_font)
                
            self.draw_instructions()
            
            # Draw model menu overlay if open
            if self.show_model_menu:
                self.draw_model_menu()
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()


def main():
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory '{MODELS_DIR}' not found!")
        print("Please ensure model files are in the models/ directory.")
        return
    
    available = discover_models()
    if not available:
        print(f"Error: No models found in '{MODELS_DIR}'!")
        print("Expected structure:")
        print("  models/resnet/model.pth")
        print("  models/protonet/model.pth")
        return
    
    app = DoodleApp()
    app.run()


if __name__ == "__main__":
    main()
