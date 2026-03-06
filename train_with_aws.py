from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

role = "arn:aws:iam::108830828338:role/SageMakerFullAccess"
output_path = "s3://dedireformer/"

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=output_path
)

# Create PyTorch Estimator
estimator = PyTorch(
    entry_point="train.py",  
    source_dir=".",  
    role=role,
    framework_version="1.12",
    py_version="py38", 
    instance_count=1,  
    instance_type="ml.c5.2xlarge",  
    output_path=output_path,  
    tensorboard_output_config=tensorboard_output_config, 
    hyperparameters={
        "model-type": "virface",  
        "learning-rate": 0.001,  
        "epochs": 100, 
        "batch-size": 128,
        "tensorboard-dir": "/opt/ml/output/tensorboard/",  
        "model-dir": "/opt/ml/model",  
    },
)

# Start training
estimator.fit()
