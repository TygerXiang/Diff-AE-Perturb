from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"
    prediction_type: str = "eps"


@dataclass(frozen=True)
class TrainingLossConfig:
    signature_gene_weight: float = 1.0
    reconstruction_loss: str = "mse"
    autofocus_gamma: float = 2.0
    direction_lambda: float = 1e-3
    lambda_semantic_align: float = 0.0
    lambda_direction_align: float = 0.0


@dataclass(frozen=True)
class GuideFeatureEncoderConfig:
    enabled: bool = False
    feature_dim: int = 0
    activity_source: str = "predicted"
    explicit_condition: bool = True
    activity_condition_dim: int = 16
    activity_scales_delta: bool = False
    architecture: str = "cnn"
    hidden_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    sequence_length: int = 26
    cnn_channels: tuple[int, ...] = (32, 32)
    cnn_kernel_size: int = 4
    cnn_dropout: float = 0.25
    lambda_strength: float = 0.0
    doser_type: str = "sigm"


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int
    n_perturbations: int
    latent_dim: int = 128
    encoder_hidden_dims: tuple[int, ...] = (512, 256)
    denoiser_hidden_dims: tuple[int, ...] = (1024, 1024)
    dropout: float = 0.0
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    losses: TrainingLossConfig = field(default_factory=TrainingLossConfig)
    guide_encoder: GuideFeatureEncoderConfig = field(default_factory=GuideFeatureEncoderConfig)
