# Training parameters
batch_size = 32
epochs = 30
learning_rate = 0.001

# Image parameters
resize_x = 160
resize_y = 160
input_channels = 3

# Model parameters
embedding_size = 128
num_classes = 5  # Will be set based on dataset

# Paths
data_path = "./data/train_data"
checkpoint_path = "./checkpoints/final_weights.pth"