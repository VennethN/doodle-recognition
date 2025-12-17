#!/usr/bin/env python3
"""
Doodle Recognition Web App - Flask-based web interface for doodle recognition
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import base64
import io
import pickle
import cv2
from abc import ABC, abstractmethod


app = Flask(__name__)

# Model paths
MODELS_DIR = "models"


# ============================================================================
# CV2SimilarityClassifier - needed for unpickling the similarity model
# ============================================================================

class CV2SimilarityClassifier:
    """
    Classifier using OpenCV's template matching and feature matching.
    This class is needed to unpickle the trained similarity model.
    """
    
    def __init__(self, method='multi'):
        self.method = method
        self.templates = {}
        self.classes_ = None
        
        if method in ['features', 'multi']:
            try:
                self.detector = cv2.SIFT_create()
                self.matcher = cv2.BFMatcher()
            except:
                self.detector = cv2.ORB_create()
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _template_match_score(self, img, template):
        img_norm = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        template_norm = cv2.normalize(template.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        scores = []
        for method in methods:
            result = cv2.matchTemplate(img_norm, template_norm, method)
            scores.append(np.max(result))
        return np.mean(scores)
    
    def _feature_match_score(self, img, template_data):
        kp_img, desc_img = self.detector.detectAndCompute(img, None)
        if desc_img is None or len(kp_img) < 4:
            return 0.0
        
        best_matches = []
        for desc_template in template_data.get('descriptors', []):
            if desc_template is None:
                continue
            try:
                if hasattr(self.matcher, 'getCrossCheck') and self.matcher.getCrossCheck():
                    matches = self.matcher.match(desc_img, desc_template)
                    matches = sorted(matches, key=lambda x: x.distance)
                    best_matches.extend(matches[:20])
                else:
                    matches = self.matcher.knnMatch(desc_img, desc_template, k=2)
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                best_matches.append(m)
            except:
                continue
        
        if len(best_matches) == 0:
            return 0.0
        return min(len(best_matches) / max(len(kp_img), 1), 1.0)
    
    def _histogram_match_score(self, img, template):
        hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_template = cv2.calcHist([template], [0], None, [256], [0, 256])
        cv2.normalize(hist_img, hist_img, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)
        
        methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
        scores = []
        for method in methods:
            score = cv2.compareHist(hist_img, hist_template, method)
            if method == cv2.HISTCMP_BHATTACHARYYA:
                score = 1.0 - min(score, 1.0)
            scores.append(score)
        return np.mean(scores)
    
    def _compute_similarity(self, img, template_data):
        template = template_data['mean']
        scores = []
        
        if self.method in ['template', 'multi']:
            scores.append(self._template_match_score(img, template))
        if self.method in ['features', 'multi']:
            scores.append(self._feature_match_score(img, template_data))
        if self.method in ['histogram', 'multi']:
            scores.append(self._histogram_match_score(img, template))
        
        return np.mean(scores) if scores else 0.0

# Global model storage
model_backends = {}
current_model_key = None
device = None


# ============================================================================
# Model Backend Classes (same as doodle_app.py)
# ============================================================================

class ModelBackend(ABC):
    """Abstract base class for model backends"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass
    
    @property
    @abstractmethod
    def all_labels(self) -> list:
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, top_k: int = 10) -> list:
        """Predict from numpy array image, return list of (label, confidence) tuples"""
        pass
    
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
        }


class ResNetBackend(ModelBackend):
    """ResNet classification model backend"""
    
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
        
        # Create and load model (matches training architecture)
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self._num_classes)
        )
        
        model_path = os.path.join(model_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        
        # Transform (3-channel to match training)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @property
    def name(self) -> str:
        return "ResNet Classification"
    
    @property
    def short_name(self) -> str:
        return "ResNet"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def all_labels(self) -> list:
        return self._all_labels
    
    def predict(self, image: np.ndarray, top_k=10) -> list:
        # Convert numpy array to PIL Image (RGB)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        img = Image.fromarray(image).convert('RGB')
        
        # Invert if needed (check mean of grayscale version)
        img_gray = np.array(img.convert('L'))
        if img_gray.mean() > 127:
            img_array = 255 - np.array(img)
            img = Image.fromarray(img_array)
        
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.idx_to_label[idx.item()]
            predictions.append((label, prob.item()))
        
        return predictions
    
    def get_info(self) -> dict:
        info = {"name": self.name, "num_classes": self._num_classes}
        model_info_path = os.path.join(self.model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                info.update(json.load(f))
        return info


class SimilarityBackend(ModelBackend):
    """OpenCV similarity matching backend"""
    
    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        self.model_dir = model_dir
        
        # Load model info
        model_info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                info = json.load(f)
            self.img_size = info.get('image_size', 64)
            self.method = info.get('method', 'multi')
            self.use_sobel = info.get('use_sobel', False)
        else:
            self.img_size = 64
            self.method = 'multi'
            self.use_sobel = False
        
        # Load label mappings
        with open(os.path.join(model_dir, "label_mappings.json"), 'r') as f:
            mappings = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in mappings['idx_to_label'].items()}
        self._all_labels = list(self.idx_to_label.values())
        self._num_classes = len(self._all_labels)
        
        # Load the classifier
        classifier_path = os.path.join(model_dir, "classifier.pkl")
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Ensure method matches
        if hasattr(self.classifier, 'method'):
            self.method = self.classifier.method
        
        # Recreate detector and matcher if needed (they don't survive pickling)
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
        return "OpenCV Similarity"
    
    @property
    def short_name(self) -> str:
        return "Similarity"
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def all_labels(self) -> list:
        return self._all_labels
    
    def predict(self, image: np.ndarray, top_k=10) -> list:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        if img_resized.mean() > 127:
            img_resized = 255 - img_resized
        
        img_uint8 = img_resized.astype(np.uint8)
        
        # Apply Sobel edge detection if model was trained with it
        if self.use_sobel:
            img_uint8 = self._apply_sobel(img_uint8)
        
        # Compute similarities
        similarities = {}
        for cls_idx in self.classifier.classes_:
            template_data = self.classifier.templates[cls_idx]
            similarity = self.classifier._compute_similarity(img_uint8, template_data)
            label = self.idx_to_label[cls_idx]
            similarities[label] = similarity
        
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        sim_values = [sim for _, sim in sorted_similarities[:top_k]]
        if not sim_values:
            return []
        
        max_sim = max(sim_values)
        exp_sims = [np.exp((sim - max_sim) * 5) for sim in sim_values]
        sum_exp = sum(exp_sims)
        
        predictions = []
        for i, (label, similarity) in enumerate(sorted_similarities[:top_k]):
            prob = exp_sims[i] / sum_exp if sum_exp > 0 else 1.0 / len(sim_values)
            predictions.append((label, float(prob)))
        
        return predictions
    
    def _apply_sobel(self, img):
        """Apply Sobel edge detection to an image.
        
        The Sobel filter detects edges by computing gradients in x and y directions.
        This is a classical computer vision technique - the kernel weights are 
        hand-designed, NOT learned like in CNNs.
        """
        # Sobel kernels for x and y gradients
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
        
        # Compute gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-255 range
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return magnitude.astype(np.uint8)
    
    def get_info(self) -> dict:
        info = {"name": self.name, "num_classes": self._num_classes}
        model_info_path = os.path.join(self.model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                info.update(json.load(f))
        return info


# ============================================================================
# Model Discovery and Loading
# ============================================================================

def discover_models() -> dict:
    """Discover available models in the models directory"""
    models_found = {}
    
    if os.path.exists(MODELS_DIR):
        for model_name in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_name)
            if not os.path.isdir(model_path):
                continue
            
            if os.path.exists(os.path.join(model_path, "model.pth")):
                models_found[model_name] = model_path
            elif os.path.exists(os.path.join(model_path, "classifier.pkl")):
                model_info_path = os.path.join(model_path, "model_info.json")
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r') as f:
                            info = json.load(f)
                        model_type = info.get('model_type')
                        if model_type == 'CV2SimilarityClassifier':
                            models_found[model_name] = model_path
                    except:
                        pass
                else:
                    models_found[model_name] = model_path
    
    return models_found


def load_model_backend(model_type: str, model_dir: str, device: torch.device) -> ModelBackend:
    """Load appropriate model backend based on type"""
    if model_type == "resnet":
        return ResNetBackend(model_dir, device)
    elif model_type == "similarity":
        return SimilarityBackend(model_dir, device)
    else:
        # Auto-detect
        if os.path.exists(os.path.join(model_dir, "classifier.pkl")):
            model_info_path = os.path.join(model_dir, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    info = json.load(f)
                classifier_type = info.get('model_type')
                if classifier_type == 'CV2SimilarityClassifier':
                    return SimilarityBackend(model_dir, device)
                else:
                    raise ValueError(f"Classifier type '{classifier_type}' not supported. Only CV2SimilarityClassifier is supported.")
            return SimilarityBackend(model_dir, device)
        elif os.path.exists(os.path.join(model_dir, "model.pth")):
            # Assume it's ResNet if model.pth exists
            return ResNetBackend(model_dir, device)
        raise ValueError(f"Unknown model type: {model_type}")


def init_models():
    """Initialize all models"""
    global model_backends, current_model_key, device
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Discover and load models
    available_models = discover_models()
    print(f"Found {len(available_models)} model(s): {list(available_models.keys())}")
    
    for model_key, model_path in available_models.items():
        try:
            print(f"  Loading {model_key}...")
            backend = load_model_backend(model_key, model_path, device)
            model_backends[model_key] = backend
            print(f"    ✓ {backend.name} ({backend.num_classes} categories)")
            
            if current_model_key is None:
                current_model_key = model_key
        except Exception as e:
            print(f"    ✗ Failed to load {model_key}: {e}")
    
    if not model_backends:
        raise RuntimeError("No models found!")
    
    print(f"\nDefault model: {model_backends[current_model_key].name}")


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_list = []
    for key, backend in model_backends.items():
        models_list.append({
            "key": key,
            "name": backend.name,
            "num_classes": backend.num_classes,
            "is_current": key == current_model_key
        })
    return jsonify({"models": models_list, "current": current_model_key})


@app.route('/api/model/<model_key>', methods=['POST'])
def set_model(model_key):
    """Set the current model"""
    global current_model_key
    if model_key in model_backends:
        current_model_key = model_key
        return jsonify({"success": True, "model": model_key})
    return jsonify({"success": False, "error": "Model not found"}), 404


@app.route('/api/model/<model_key>/info', methods=['GET'])
def get_model_info(model_key):
    """Get detailed model information"""
    if model_key in model_backends:
        return jsonify(model_backends[model_key].get_info())
    return jsonify({"error": "Model not found"}), 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction from canvas image data"""
    global current_model_key
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Use specified model or current
    model_key = data.get('model', current_model_key)
    if model_key not in model_backends:
        return jsonify({"error": "Model not found"}), 404
    
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        
        # Create white background and paste image
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        
        img_array = np.array(background)
        
        # Get predictions
        backend = model_backends[model_key]
        predictions = backend.predict(img_array, top_k=10)
        
        return jsonify({
            "predictions": [{"label": label, "confidence": conf} for label, conf in predictions],
            "model": model_key
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories for the current model"""
    if current_model_key and current_model_key in model_backends:
        return jsonify({
            "categories": model_backends[current_model_key].all_labels,
            "model": current_model_key
        })
    return jsonify({"error": "No model loaded"}), 500


@app.route('/api/challenge', methods=['GET'])
def get_challenge():
    """Get a random category for challenge mode"""
    import random
    if current_model_key and current_model_key in model_backends:
        categories = model_backends[current_model_key].all_labels
        word = random.choice(categories)
        return jsonify({
            "word": word,
            "model": current_model_key,
            "total_categories": len(categories)
        })
    return jsonify({"error": "No model loaded"}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Initialize models before starting server
    init_models()
    
    print("\n" + "="*50)
    print("Starting Doodle Recognition Web App")
    print("Open http://localhost:5001 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5001)

