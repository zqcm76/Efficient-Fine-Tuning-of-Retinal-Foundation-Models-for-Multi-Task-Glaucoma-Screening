### Cost vs accuracy: scratch baseline vs LoRA-RETFound

| model                   |   params_total_M |   params_trainable_M |   train_hours |   peak_train_mem_GB |   infer_ms_per_img |   infer_peak_mem_GB |   indomain_auc |   indomain_disc_dice |   indomain_cup_dice |   indomain_vcdr_mae |   mean_external_auc |   auc_drop_external |
|:------------------------|-----------------:|---------------------:|--------------:|--------------------:|-------------------:|--------------------:|---------------:|---------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| ResNet34+UNet (scratch) |             24.4 |                 24.4 |           3.2 |                 7.1 |                 12 |                 1.8 |         0.8986 |               0.9728 |              0.9002 |              0.0504 |              0.7953 |              0.1033 |
| LoRA-RETFound           |            304   |                  4.7 |           6.5 |                11.4 |                 41 |                 3.2 |         1      |               0.9892 |              0.9741 |              0.0138 |              0.8405 |              0.1595 |
