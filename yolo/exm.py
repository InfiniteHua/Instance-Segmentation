import torch
print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
