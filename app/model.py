from insightface.app import FaceAnalysis
import onnxruntime as ort
import logging
import os 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import mediapipe as mp
from insightface.model_zoo import model_zoo
from email_feedback import EmailFeedback

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

class Model:
    def __init__(self):
        self.MODEL_NAME="antelopev2"
        MODEL_CACHE_DIR = "./models/.insightface"
        self.model_root=MODEL_CACHE_DIR
        self.DET_SIZE=(384, 384) 
        self.app=None
    def initialize_insightface(self):
        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4  # Utilize multiple cores
            session_options.inter_op_num_threads = 4
            
            # Initialize InsightFace with GPU support, fallback to CPU
            self.app = FaceAnalysis(
                name=self.MODEL_NAME,
                root=self.model_root,
                providers=['CUDAExecutionProvider',],
                session_options=session_options,
                download=False
            )
            self.app.prepare(ctx_id=-1, det_size=self.DET_SIZE)
            logging.info("INFO: InsightFace model initialized successfully")
            EmailFeedback.compose_email("Success", "InsightFace model initialized successfully")
            return self.app
        
        except Exception as e:
            error_msg = f"Insightface initialization failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e
    def initialize_mediapipe(self):
        try:
            self.facemesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info("INFO: Mediapipe model initialized successfully")
            EmailFeedback.compose_email("Success", "Mediapipe model initialized successfully")
            return self.facemesh
    
        except Exception as e:
            error_msg = f"Mediapipe initialization failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e

    def FineTuner(self, num_epochs, batch_size=32, learning_rate=0.001, device=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def load_model(self):
        try:
            self.model = model_zoo.get_model(self.MODEL_NAME, root=self.model_root)
            self.model.to(self.device)
            self.model.train()
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name}: {str(e)}")

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = FaceDataset(root_dir=self.root_dir, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        if not self.model:
            self.load_model()
        if not self.train_loader:
            self.prepare_data()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy}%')

    def save_model(self, save_path='/kaggle/working/fine_tuned_model.pth'):
        if self.model:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save. Please load or train a model first.")

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.img_labels = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                self.img_labels.append((os.path.join(cls_path, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
