# Fine-tune Yibo's prior on yibo.yaml — November 17, 2026

Initialize from `pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth`
weights with a fresh optimizer and epoch, and train using the current `yibo.yaml`
recipe. All robot, observation, cloth physics, randomization, reward and episode
check settings are inherited unchanged, including 2 mm physical plates, 16 mm
observed thickness, 12 mm cloth contact offset and 4–10 mm rest-offset DR. Both
current reset checks and the fold/release objective remain enabled.

The environment batch is reduced from 24,576 to **12,288** for one 48 GB GPU, matching
the other runs. Six SAPG blocks use 2,048 environments each. Periodic checkpoints
are saved every 10 epochs; best-checkpoint saving starts after epoch 1.

W&B project: `rigidClothDR`; group: `nov17_yibo_prior_finetune`. Interactive viewer
captures are enabled per cloth design. Existing runs are retained.

```bash
experiments/nov17_yibo_prior_finetune/submit.sh
```
