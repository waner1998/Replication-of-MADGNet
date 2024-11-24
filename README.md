# Model Output Issue with Batch Size

## Description

I observed a significant difference in the model's output when using different batch sizes. Specifically:

- **Left**: Output with `batchsize=16`.
- **Right**: Output with `batchsize=6`.

I don't know why this discrepancy occurs. Any insights or suggestions are appreciated.

![Comparison of outputs with different batch sizes](https://github.com/user-attachments/assets/09a4e547-b50a-4518-b6bc-c42508b9c24f)

python train.py --train_image_path {train_image_path} --train_mask_path {train_mask_path} --save_path {save_path} --epoch 50 --gpu_choose 0 --batch_size 16 --num_workers 4

python test.py --checkpoint {checkpoint_path} --test_image_path {img_path} --test_gt_path {label_path} --save_path {save_path} --device 0

python eval.py --dataset_name {dataset_name} --pred_path {pred_path} --gt_path {gt_path}
