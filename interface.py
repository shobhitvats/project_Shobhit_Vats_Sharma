from model import FaceRecognitionModel as TheModel
from train import train_model as the_trainer
from predict import classify_faces as the_predictor
from dataset import FaceDataset as TheDataset
from dataset import create_dataloader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs