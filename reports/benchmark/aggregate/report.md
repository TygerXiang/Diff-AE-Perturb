# gse132080 six-metric benchmark

- methods: control_mean, cpa, diffae, diffae_delta_recompute, diffae_delta_x50, fix_eps_batchmean, fix_eps_percell, fix_x0_batchmean, fix_x0_percell, gears, legacy_x0_percell, linear_pca, linear_ridge, scgen, squidiff, squidiff_repro, squidiff_repro_clip, squidiff_repro_noclamp, train_mean
- seeds: 7
- metrics: mse, pcc_delta, e_distance, wasserstein, kl_divergence, common_degs

## Mean over evaluated conditions

### common_degs@cutoff (higher is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | linear_ridge | 0.267264 |  | 1 |
| 2 | linear_pca | 0.204616 |  | 1 |
| 3 | scgen | 0.198495 |  | 1 |
| 4 | gears | 0.153786 |  | 1 |
| 5 | fix_x0_percell | 0.129396 |  | 1 |
| 6 | fix_x0_batchmean | 0.123950 |  | 1 |
| 7 | diffae_delta_recompute | 0.118034 |  | 1 |
| 8 | squidiff_repro | 0.070000 |  | 1 |
| 9 | squidiff | 0.069999 |  | 1 |
| 10 | diffae_delta_x50 | 0.056217 |  | 1 |
| 11 | cpa | 0.051805 |  | 1 |
| 12 | squidiff_repro_clip | 0.038254 |  | 1 |
| 13 | train_mean | 0.036538 |  | 1 |
| 14 | diffae | 0.036309 |  | 1 |
| 15 | legacy_x0_percell | 0.027667 |  | 1 |
| 16 | fix_eps_batchmean | 0.011718 |  | 1 |
| 17 | fix_eps_percell | 0.008085 |  | 1 |
| 18 | squidiff_repro_noclamp | 0.004040 |  | 1 |
| 19 | control_mean | 0.000000 |  | 1 |

### e_distance@100 (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | gears | 0.458947 |  | 1 |
| 2 | linear_ridge | 0.513460 |  | 1 |
| 3 | diffae_delta_recompute | 0.775817 |  | 1 |
| 4 | fix_x0_percell | 0.898848 |  | 1 |
| 5 | scgen | 0.908575 |  | 1 |
| 6 | linear_pca | 0.993181 |  | 1 |
| 7 | fix_x0_batchmean | 1.069676 |  | 1 |
| 8 | train_mean | 1.578395 |  | 1 |
| 9 | control_mean | 1.802712 |  | 1 |
| 10 | diffae_delta_x50 | 1.965989 |  | 1 |
| 11 | legacy_x0_percell | 1.991308 |  | 1 |
| 12 | diffae | 2.016981 |  | 1 |
| 13 | cpa | 3.955410 |  | 1 |
| 14 | squidiff_repro_clip | 14.988506 |  | 1 |
| 15 | squidiff | 352.566380 |  | 1 |
| 16 | fix_eps_percell | 372.083588 |  | 1 |
| 17 | fix_eps_batchmean | 385.638328 |  | 1 |
| 18 | squidiff_repro | 435.705192 |  | 1 |
| 19 | squidiff_repro_noclamp | 466.065835 |  | 1 |

### e_distance@all (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | linear_ridge | 0.477923 |  | 1 |
| 2 | gears | 0.563596 |  | 1 |
| 3 | diffae_delta_recompute | 0.854295 |  | 1 |
| 4 | linear_pca | 0.895351 |  | 1 |
| 5 | scgen | 1.221925 |  | 1 |
| 6 | fix_x0_percell | 1.247116 |  | 1 |
| 7 | train_mean | 1.289641 |  | 1 |
| 8 | control_mean | 1.440864 |  | 1 |
| 9 | fix_x0_batchmean | 1.508640 |  | 1 |
| 10 | legacy_x0_percell | 1.764229 |  | 1 |
| 11 | diffae | 1.800163 |  | 1 |
| 12 | diffae_delta_x50 | 2.014221 |  | 1 |
| 13 | cpa | 4.935972 |  | 1 |
| 14 | squidiff_repro_clip | 17.907049 |  | 1 |
| 15 | squidiff | 606.330711 |  | 1 |
| 16 | fix_eps_percell | 735.400340 |  | 1 |
| 17 | fix_eps_batchmean | 768.441372 |  | 1 |
| 18 | squidiff_repro | 777.213161 |  | 1 |
| 19 | squidiff_repro_noclamp | 888.121230 |  | 1 |

### kl_divergence@100 (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | fix_x0_batchmean | 12.477912 |  | 1 |
| 2 | fix_x0_percell | 12.503866 |  | 1 |
| 3 | scgen | 12.526211 |  | 1 |
| 4 | diffae_delta_recompute | 12.745897 |  | 1 |
| 5 | linear_ridge | 13.091425 |  | 1 |
| 6 | linear_pca | 13.191699 |  | 1 |
| 7 | train_mean | 13.302791 |  | 1 |
| 8 | control_mean | 13.348100 |  | 1 |
| 9 | diffae_delta_x50 | 13.498325 |  | 1 |
| 10 | diffae | 13.821154 |  | 1 |
| 11 | legacy_x0_percell | 13.844241 |  | 1 |
| 12 | cpa | 16.557743 |  | 1 |
| 13 | squidiff_repro_clip | 17.156833 |  | 1 |
| 14 | gears | 20.325686 |  | 1 |
| 15 | squidiff | 25.089771 |  | 1 |
| 16 | squidiff_repro | 26.011909 |  | 1 |
| 17 | fix_eps_percell | 27.590441 |  | 1 |
| 18 | fix_eps_batchmean | 27.749022 |  | 1 |
| 19 | squidiff_repro_noclamp | 27.917804 |  | 1 |

### kl_divergence@all (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | scgen | 35.495660 |  | 1 |
| 2 | fix_x0_batchmean | 36.453580 |  | 1 |
| 3 | fix_x0_percell | 36.758542 |  | 1 |
| 4 | legacy_x0_percell | 37.307144 |  | 1 |
| 5 | diffae | 37.431762 |  | 1 |
| 6 | diffae_delta_x50 | 37.725979 |  | 1 |
| 7 | diffae_delta_recompute | 37.744988 |  | 1 |
| 8 | cpa | 38.437993 |  | 1 |
| 9 | linear_ridge | 39.777511 |  | 1 |
| 10 | control_mean | 39.784120 |  | 1 |
| 11 | train_mean | 39.791117 |  | 1 |
| 12 | linear_pca | 39.862383 |  | 1 |
| 13 | gears | 41.706032 |  | 1 |
| 14 | squidiff_repro_clip | 49.528323 |  | 1 |
| 15 | squidiff | 56.640014 |  | 1 |
| 16 | squidiff_repro | 57.936355 |  | 1 |
| 17 | squidiff_repro_noclamp | 60.313905 |  | 1 |
| 18 | fix_eps_percell | 60.549893 |  | 1 |
| 19 | fix_eps_batchmean | 60.712296 |  | 1 |

### mse@100 (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | gears | 0.020826 |  | 1 |
| 2 | scgen | 0.021268 |  | 1 |
| 3 | linear_ridge | 0.024521 |  | 1 |
| 4 | fix_x0_percell | 0.039266 |  | 1 |
| 5 | diffae_delta_recompute | 0.040743 |  | 1 |
| 6 | fix_x0_batchmean | 0.043085 |  | 1 |
| 7 | linear_pca | 0.064553 |  | 1 |
| 8 | train_mean | 0.120322 |  | 1 |
| 9 | cpa | 0.135282 |  | 1 |
| 10 | diffae_delta_x50 | 0.136338 |  | 1 |
| 11 | legacy_x0_percell | 0.140407 |  | 1 |
| 12 | control_mean | 0.140728 |  | 1 |
| 13 | diffae | 0.143308 |  | 1 |
| 14 | squidiff_repro_clip | 1.870213 |  | 1 |
| 15 | squidiff_repro_noclamp | 63.641286 |  | 1 |
| 16 | fix_eps_batchmean | 83.589559 |  | 1 |
| 17 | fix_eps_percell | 84.220563 |  | 1 |
| 18 | squidiff | 664.127340 |  | 1 |
| 19 | squidiff_repro | 1005.041305 |  | 1 |

### mse@all (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | scgen | 0.002984 |  | 1 |
| 2 | linear_ridge | 0.003488 |  | 1 |
| 3 | gears | 0.004495 |  | 1 |
| 4 | diffae_delta_recompute | 0.005458 |  | 1 |
| 5 | fix_x0_percell | 0.005937 |  | 1 |
| 6 | fix_x0_batchmean | 0.006580 |  | 1 |
| 7 | linear_pca | 0.008478 |  | 1 |
| 8 | train_mean | 0.013996 |  | 1 |
| 9 | cpa | 0.015812 |  | 1 |
| 10 | control_mean | 0.016079 |  | 1 |
| 11 | legacy_x0_percell | 0.016121 |  | 1 |
| 12 | diffae | 0.016483 |  | 1 |
| 13 | diffae_delta_x50 | 0.018585 |  | 1 |
| 14 | squidiff_repro_clip | 0.342666 |  | 1 |
| 15 | squidiff_repro_noclamp | 24.037695 |  | 1 |
| 16 | fix_eps_percell | 32.772845 |  | 1 |
| 17 | fix_eps_batchmean | 32.863195 |  | 1 |
| 18 | squidiff | 188.787697 |  | 1 |
| 19 | squidiff_repro | 306.611844 |  | 1 |

### pcc_delta@100 (higher is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | gears | 0.895589 |  | 1 |
| 2 | diffae_delta_recompute | 0.884925 |  | 1 |
| 3 | linear_ridge | 0.867429 |  | 1 |
| 4 | scgen | 0.863893 |  | 1 |
| 5 | fix_x0_percell | 0.823224 |  | 1 |
| 6 | fix_x0_batchmean | 0.807111 |  | 1 |
| 7 | linear_pca | 0.677450 |  | 1 |
| 8 | train_mean | 0.465249 |  | 1 |
| 9 | squidiff_repro_clip | 0.337250 |  | 1 |
| 10 | diffae_delta_x50 | 0.240854 |  | 1 |
| 11 | cpa | 0.191665 |  | 1 |
| 12 | squidiff_repro_noclamp | 0.077157 |  | 1 |
| 13 | fix_eps_percell | 0.074539 |  | 1 |
| 14 | legacy_x0_percell | 0.066973 |  | 1 |
| 15 | fix_eps_batchmean | 0.047972 |  | 1 |
| 16 | diffae | 0.028314 |  | 1 |
| 17 | control_mean | -0.012392 |  | 1 |
| 18 | squidiff_repro | -0.092636 |  | 1 |
| 19 | squidiff | -0.121157 |  | 1 |

### pcc_delta@all (higher is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | linear_ridge | 0.803045 |  | 1 |
| 2 | scgen | 0.795696 |  | 1 |
| 3 | diffae_delta_recompute | 0.794822 |  | 1 |
| 4 | gears | 0.744061 |  | 1 |
| 5 | fix_x0_percell | 0.729793 |  | 1 |
| 6 | fix_x0_batchmean | 0.701232 |  | 1 |
| 7 | linear_pca | 0.557792 |  | 1 |
| 8 | train_mean | 0.340420 |  | 1 |
| 9 | diffae_delta_x50 | 0.192955 |  | 1 |
| 10 | cpa | 0.153094 |  | 1 |
| 11 | squidiff_repro_clip | 0.131164 |  | 1 |
| 12 | squidiff_repro_noclamp | 0.041848 |  | 1 |
| 13 | fix_eps_percell | 0.041029 |  | 1 |
| 14 | legacy_x0_percell | 0.030887 |  | 1 |
| 15 | fix_eps_batchmean | 0.027719 |  | 1 |
| 16 | diffae | 0.023067 |  | 1 |
| 17 | squidiff_repro | 0.022194 |  | 1 |
| 18 | squidiff | 0.012438 |  | 1 |
| 19 | control_mean | -0.011666 |  | 1 |

### wasserstein@100 (lower is better)

| rank | method | mean | sd | n |
|---|---|---|---|---|
| 1 | scgen | 27.299535 |  | 1 |
| 2 | fix_x0_batchmean | 28.011030 |  | 1 |
| 3 | fix_x0_percell | 28.389225 |  | 1 |
| 4 | diffae_delta_recompute | 32.707175 |  | 1 |
| 5 | gears | 35.580166 |  | 1 |
| 6 | cpa | 37.033642 |  | 1 |
| 7 | linear_ridge | 37.535861 |  | 1 |
| 8 | linear_pca | 41.749075 |  | 1 |
| 9 | diffae_delta_x50 | 42.980988 |  | 1 |
| 10 | legacy_x0_percell | 43.715528 |  | 1 |
| 11 | diffae | 43.882389 |  | 1 |
| 12 | train_mean | 47.505334 |  | 1 |
| 13 | control_mean | 49.622519 |  | 1 |
| 14 | squidiff_repro_clip | 276.422846 |  | 1 |
| 15 | squidiff | 151812.482813 |  | 1 |
| 16 | squidiff_repro | 233915.982292 |  | 1 |
| 17 | fix_eps_percell | 394557.043750 |  | 1 |
| 18 | fix_eps_batchmean | 425288.368750 |  | 1 |
| 19 | squidiff_repro_noclamp | 629770.379167 |  | 1 |

## Top-1 counts across metric/budget cells

| method | wins |
|---|---|
| gears | 3 |
| linear_ridge | 3 |
| scgen | 3 |
| fix_x0_batchmean | 1 |
