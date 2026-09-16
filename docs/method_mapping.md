# Method-to-Code Mapping

This note maps the simplified conditional DDIM method in `methods_revised.md`
to implementation files.

| Methods item | Implementation |
| --- | --- |
| Semantic encoder `E_phi(x)` | `src/perturbdiff/modules.py::SemanticEncoder` |
| Perturbation delta table `W[g]` | `src/perturbdiff/modules.py::PerturbationModule` |
| Forward diffusion | `src/perturbdiff/diffusion.py::GaussianDiffusion.q_sample` |
| Conditional denoising network | `src/perturbdiff/modules.py::ConditionalDenoiserMLP` |
| Diffusion reconstruction loss | `src/perturbdiff/losses.py::weighted_mse_loss` |
| Signature-gene reconstruction weighting | `src/perturbdiff/data.py::build_signature_gene_weights` and `src/perturbdiff/model.py` |
| Counterfactual DDIM generation | `PerturbDiffusionModel.sample_counterfactual` and `GaussianDiffusion.ddim_sample` |
| sgRNA-level label preservation | `src/perturbdiff/data.py::encode_perturbation_labels(..., target_strategy="raw")` |

The diffusion clock is represented as integer timesteps `0..T`, with
`alpha_bar[0] = 1`. Training samples `t in {1, ..., T}` and DDIM sampling returns
the generated expression vector at the explicit endpoint `tau_0 = 0`.

The current denoiser is an MLP for vector-valued expression profiles. This is a
small baseline kept intentionally simple before adding graph structure,
attention, sgRNA-strength modeling, or other perturbation-specific machinery.
