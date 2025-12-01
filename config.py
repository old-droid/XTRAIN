"""
CPUWARP-ML Configuration Manager
===============================

Centralized configuration management with .env file support
Handles model parameters, dataset paths, and multimodal settings
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"Warning: {env_path} not found. Using default configuration.")
        return env_vars
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle variable substitution
                if '${' in value:
                    for var_name, var_value in env_vars.items():
                        value = value.replace(f'${{{var_name}}}', var_value)
                
                env_vars[key] = value
                # Also set as environment variable for subprocess access
                os.environ[key] = value
    
    return env_vars

def get_env_value(key: str, default: Any = None, dtype: type = str) -> Any:
    """Get environment variable with type conversion"""
    value = os.environ.get(key, default)
    
    if value is None:
        return None
    
    if dtype == bool:
        return str(value).lower() in ('true', '1', 'yes', 'on')
    elif dtype == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    elif dtype == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    elif dtype == list:
        return str(value).split(',') if value else []
    else:
        return str(value)

@dataclass
class DatasetConfig:
    """Dataset configuration from environment"""
    root: str = field(default_factory=lambda: get_env_value('DATASET_ROOT', './datasets'))
    
    # Image datasets
    imagenet_path: str = field(default_factory=lambda: get_env_value('IMAGENET_PATH', './datasets/imagenet'))
    cifar10_path: str = field(default_factory=lambda: get_env_value('CIFAR10_PATH', './datasets/cifar-10-batches-py'))
    cifar100_path: str = field(default_factory=lambda: get_env_value('CIFAR100_PATH', './datasets/cifar-100-python'))
    mnist_path: str = field(default_factory=lambda: get_env_value('MNIST_PATH', './datasets/mnist'))
    coco_path: str = field(default_factory=lambda: get_env_value('COCO_PATH', './datasets/coco'))
    
    # Text datasets
    wikitext_path: str = field(default_factory=lambda: get_env_value('WIKITEXT_PATH', './datasets/wikitext'))
    bookcorpus_path: str = field(default_factory=lambda: get_env_value('BOOKCORPUS_PATH', './datasets/bookcorpus'))
    
    # Audio datasets
    librispeech_path: str = field(default_factory=lambda: get_env_value('LIBRISPEECH_PATH', './datasets/librispeech'))
    common_voice_path: str = field(default_factory=lambda: get_env_value('COMMON_VOICE_PATH', './datasets/common_voice'))
    
    # Multimodal datasets
    vqa_path: str = field(default_factory=lambda: get_env_value('VQA_PATH', './datasets/vqa'))
    flickr30k_path: str = field(default_factory=lambda: get_env_value('FLICKR30K_PATH', './datasets/flickr30k'))
    
    # Data pipeline settings
    cache_dataset: bool = field(default_factory=lambda: get_env_value('CACHE_DATASET', True, bool))
    cache_dir: str = field(default_factory=lambda: get_env_value('DATASET_CACHE_DIR', './cache'))
    num_workers: int = field(default_factory=lambda: get_env_value('PREPROCESS_NUM_WORKERS', 8, int))
    shuffle_buffer_size: int = field(default_factory=lambda: get_env_value('SHUFFLE_BUFFER_SIZE', 10000, int))

@dataclass  
class ModelConfig:
    """Model architecture configuration"""
    # CNN Configuration
    cnn_input_size: int = field(default_factory=lambda: get_env_value('CNN_INPUT_SIZE', 224, int))
    cnn_input_channels: int = field(default_factory=lambda: get_env_value('CNN_INPUT_CHANNELS', 3, int))
    cnn_num_classes: int = field(default_factory=lambda: get_env_value('CNN_NUM_CLASSES', 1000, int))
    cnn_architecture: str = field(default_factory=lambda: get_env_value('CNN_ARCHITECTURE', 'resnet'))
    cnn_depth: int = field(default_factory=lambda: get_env_value('CNN_DEPTH', 50, int))
    
    # LLM Configuration
    llm_vocab_size: int = field(default_factory=lambda: get_env_value('LLM_VOCAB_SIZE', 50000, int))
    llm_d_model: int = field(default_factory=lambda: get_env_value('LLM_D_MODEL', 768, int))
    llm_num_heads: int = field(default_factory=lambda: get_env_value('LLM_NUM_HEADS', 12, int))
    llm_num_layers: int = field(default_factory=lambda: get_env_value('LLM_NUM_LAYERS', 12, int))
    llm_d_ff: int = field(default_factory=lambda: get_env_value('LLM_D_FF', 3072, int))
    llm_max_seq_len: int = field(default_factory=lambda: get_env_value('LLM_MAX_SEQ_LEN', 1024, int))
    
    # Vision Transformer
    vit_patch_size: int = field(default_factory=lambda: get_env_value('VIT_PATCH_SIZE', 16, int))
    vit_image_size: int = field(default_factory=lambda: get_env_value('VIT_IMAGE_SIZE', 224, int))
    vit_d_model: int = field(default_factory=lambda: get_env_value('VIT_D_MODEL', 768, int))
    vit_num_heads: int = field(default_factory=lambda: get_env_value('VIT_NUM_HEADS', 12, int))
    vit_num_layers: int = field(default_factory=lambda: get_env_value('VIT_NUM_LAYERS', 12, int))
    vit_mlp_dim: int = field(default_factory=lambda: get_env_value('VIT_MLP_DIM', 3072, int))
    
    # Audio Models
    audio_sample_rate: int = field(default_factory=lambda: get_env_value('AUDIO_SAMPLE_RATE', 16000, int))
    audio_n_mels: int = field(default_factory=lambda: get_env_value('AUDIO_N_MELS', 80, int))
    audio_n_fft: int = field(default_factory=lambda: get_env_value('AUDIO_N_FFT', 1024, int))
    audio_hop_length: int = field(default_factory=lambda: get_env_value('AUDIO_HOP_LENGTH', 256, int))
    audio_max_length: int = field(default_factory=lambda: get_env_value('AUDIO_MAX_LENGTH', 16000, int))

@dataclass
class MultimodalConfig:
    """Multimodal model configuration"""
    enabled: bool = field(default_factory=lambda: get_env_value('ENABLE_MULTIMODAL', True, bool))
    
    # Vision-Language Model
    vlm_vision_encoder: str = field(default_factory=lambda: get_env_value('VLM_VISION_ENCODER', 'resnet50'))
    vlm_text_encoder: str = field(default_factory=lambda: get_env_value('VLM_TEXT_ENCODER', 'bert_base'))
    vlm_fusion_method: str = field(default_factory=lambda: get_env_value('VLM_FUSION_METHOD', 'concat'))
    vlm_hidden_dim: int = field(default_factory=lambda: get_env_value('VLM_HIDDEN_DIM', 512, int))
    
    # Audio-Visual Model
    av_audio_encoder: str = field(default_factory=lambda: get_env_value('AV_AUDIO_ENCODER', 'wav2vec2'))
    av_vision_encoder: str = field(default_factory=lambda: get_env_value('AV_VISION_ENCODER', 'resnet18'))
    av_fusion_dim: int = field(default_factory=lambda: get_env_value('AV_FUSION_DIM', 256, int))
    
    # Cross-Modal Attention
    cross_modal_attention: bool = field(default_factory=lambda: get_env_value('CROSS_MODAL_ATTENTION', True, bool))
    cross_modal_heads: int = field(default_factory=lambda: get_env_value('CROSS_MODAL_HEADS', 8, int))
    cross_modal_layers: int = field(default_factory=lambda: get_env_value('CROSS_MODAL_LAYERS', 4, int))
    
    # Fusion Strategy
    fusion_method: str = field(default_factory=lambda: get_env_value('MULTIMODAL_FUSION', 'late'))
    modality_dropout: float = field(default_factory=lambda: get_env_value('MODALITY_DROPOUT', 0.1, float))

@dataclass
class TrainingConfig:
    """Training configuration"""
    # CNN Training
    cnn_batch_size: int = field(default_factory=lambda: get_env_value('CNN_BATCH_SIZE', 32, int))
    cnn_learning_rate: float = field(default_factory=lambda: get_env_value('CNN_LEARNING_RATE', 0.001, float))
    cnn_epochs: int = field(default_factory=lambda: get_env_value('CNN_EPOCHS', 100, int))
    cnn_weight_decay: float = field(default_factory=lambda: get_env_value('CNN_WEIGHT_DECAY', 1e-4, float))
    
    # LLM Training
    llm_batch_size: int = field(default_factory=lambda: get_env_value('LLM_BATCH_SIZE', 16, int))
    llm_learning_rate: float = field(default_factory=lambda: get_env_value('LLM_LEARNING_RATE', 5e-4, float))
    llm_epochs: int = field(default_factory=lambda: get_env_value('LLM_EPOCHS', 10, int))
    llm_warmup_steps: int = field(default_factory=lambda: get_env_value('LLM_WARMUP_STEPS', 4000, int))
    
    # General Training
    mixed_precision: bool = field(default_factory=lambda: get_env_value('MIXED_PRECISION', True, bool))
    gradient_clipping: float = field(default_factory=lambda: get_env_value('GRADIENT_CLIPPING', 1.0, float))
    dropout_rate: float = field(default_factory=lambda: get_env_value('DROPOUT_RATE', 0.1, float))
    label_smoothing: float = field(default_factory=lambda: get_env_value('LABEL_SMOOTHING', 0.1, float))
    
    # Data Augmentation
    data_augmentation: bool = field(default_factory=lambda: get_env_value('DATA_AUGMENTATION', True, bool))
    random_crop: bool = field(default_factory=lambda: get_env_value('RANDOM_CROP', True, bool))
    random_flip: bool = field(default_factory=lambda: get_env_value('RANDOM_FLIP', True, bool))
    color_jitter: bool = field(default_factory=lambda: get_env_value('COLOR_JITTER', True, bool))
    mixup_alpha: float = field(default_factory=lambda: get_env_value('MIXUP_ALPHA', 0.2, float))
    cutmix_alpha: float = field(default_factory=lambda: get_env_value('CUTMIX_ALPHA', 1.0, float))
    
    # Evaluation
    eval_every_n_epochs: int = field(default_factory=lambda: get_env_value('EVAL_EVERY_N_EPOCHS', 1, int))
    eval_batch_size: int = field(default_factory=lambda: get_env_value('EVAL_BATCH_SIZE', 64, int))
    save_best_model: bool = field(default_factory=lambda: get_env_value('SAVE_BEST_MODEL', True, bool))
    early_stopping_patience: int = field(default_factory=lambda: get_env_value('EARLY_STOPPING_PATIENCE', 10, int))
    metric_for_best_model: str = field(default_factory=lambda: get_env_value('METRIC_FOR_BEST_MODEL', 'accuracy'))

@dataclass
class WARPConfig:
    """WARP Scheduler configuration"""
    compute_threads: Union[str, int] = field(default_factory=lambda: get_env_value('WARP_COMPUTE_THREADS', 'auto'))
    memory_threads: int = field(default_factory=lambda: get_env_value('WARP_MEMORY_THREADS', 4, int))
    cache_allocation: float = field(default_factory=lambda: get_env_value('WARP_CACHE_ALLOCATION', 0.8, float))
    enable_profiling: bool = field(default_factory=lambda: get_env_value('WARP_ENABLE_PROFILING', True, bool))
    adaptive_scheduling: bool = field(default_factory=lambda: get_env_value('WARP_ADAPTIVE_SCHEDULING', True, bool))
    
    # Memory Management
    memory_efficient: bool = field(default_factory=lambda: get_env_value('MEMORY_EFFICIENT', True, bool))
    gradient_checkpointing: bool = field(default_factory=lambda: get_env_value('GRADIENT_CHECKPOINTING', False, bool))
    max_memory_gb: int = field(default_factory=lambda: get_env_value('MAX_MEMORY_GB', 16, int))

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = field(default_factory=lambda: get_env_value('LOG_LEVEL', 'INFO'))
    log_file: str = field(default_factory=lambda: get_env_value('LOG_FILE', './logs/cpuwarp_ml.log'))
    tensorboard_dir: str = field(default_factory=lambda: get_env_value('TENSORBOARD_DIR', './runs'))
    save_checkpoint_every: int = field(default_factory=lambda: get_env_value('SAVE_CHECKPOINT_EVERY', 1000, int))
    max_checkpoints: int = field(default_factory=lambda: get_env_value('MAX_CHECKPOINTS', 5, int))
    
    # Development
    debug_mode: bool = field(default_factory=lambda: get_env_value('DEBUG_MODE', False, bool))
    profile_memory: bool = field(default_factory=lambda: get_env_value('PROFILE_MEMORY', False, bool))
    profile_compute: bool = field(default_factory=lambda: get_env_value('PROFILE_COMPUTE', False, bool))

class CPUWarpMLConfig:
    """Main configuration class for CPUWARP-ML"""
    
    def __init__(self, env_file: str = ".env"):
        # Load environment variables
        self.env_vars = load_env_file(env_file)
        
        # Initialize configuration sections
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.multimodal = MultimodalConfig()
        self.training = TrainingConfig()
        self.warp = WARPConfig()
        self.logging = LoggingConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self._create_directories()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(os.path.dirname(self.logging.log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logging.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('CPUWarpML')
        self.logger.info("CPUWARP-ML configuration loaded successfully")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.dataset.root,
            self.dataset.cache_dir,
            os.path.dirname(self.logging.log_file),
            self.logging.tensorboard_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get path for a specific dataset"""
        dataset_paths = {
            'imagenet': self.dataset.imagenet_path,
            'cifar10': self.dataset.cifar10_path,
            'cifar100': self.dataset.cifar100_path,
            'mnist': self.dataset.mnist_path,
            'coco': self.dataset.coco_path,
            'wikitext': self.dataset.wikitext_path,
            'bookcorpus': self.dataset.bookcorpus_path,
            'librispeech': self.dataset.librispeech_path,
            'common_voice': self.dataset.common_voice_path,
            'vqa': self.dataset.vqa_path,
            'flickr30k': self.dataset.flickr30k_path,
        }
        
        path = dataset_paths.get(dataset_name.lower())
        if path is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return path
    
    def is_multimodal_enabled(self) -> bool:
        """Check if multimodal capabilities are enabled"""
        return self.multimodal.enabled
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type"""
        if model_type.lower() == 'cnn':
            return {
                'input_size': self.model.cnn_input_size,
                'input_channels': self.model.cnn_input_channels,
                'num_classes': self.model.cnn_num_classes,
                'architecture': self.model.cnn_architecture,
                'depth': self.model.cnn_depth,
                'batch_size': self.training.cnn_batch_size,
                'learning_rate': self.training.cnn_learning_rate,
                'epochs': self.training.cnn_epochs,
                'weight_decay': self.training.cnn_weight_decay,
            }
        elif model_type.lower() == 'llm':
            return {
                'vocab_size': self.model.llm_vocab_size,
                'd_model': self.model.llm_d_model,
                'num_heads': self.model.llm_num_heads,
                'num_layers': self.model.llm_num_layers,
                'd_ff': self.model.llm_d_ff,
                'max_seq_len': self.model.llm_max_seq_len,
                'batch_size': self.training.llm_batch_size,
                'learning_rate': self.training.llm_learning_rate,
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)