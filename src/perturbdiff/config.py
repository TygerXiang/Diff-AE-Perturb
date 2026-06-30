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
class ModelConfig:
    n_genes: int
    n_perturbations: int
    latent_dim: int = 128
    encoder_hidden_dims: tuple[int, ...] = (512, 256)
    denoiser_hidden_dims: tuple[int, ...] = (1024, 1024)
    dropout: float = 0.0
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    losses: TrainingLossConfig = field(default_factory=TrainingLossConfig)
