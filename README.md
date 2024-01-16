# Frequency-based Boundary-guided Attention Network for Polyp Segmentation from Colonoscopy Images (Elsevier CMPB2024 Submission)

# Code Usage

STEP1. Download Github Code

STEP2. Download polyp segmentation dataset in following [link](https://drive.google.com/file/d/1qv-V8kMZolNdBOruzRLO_9UshhWGohLi/view?usp=sharing).

STEP3. Move dataset into folder 'dataset/BioMedicalDataset'

## For evaluating model, 

STEP4-1. Download model pre-trained weights in following [link](https://drive.google.com/file/d/1GLhkmDuzxN_MQpOo2rag2m9BwZK91hlE/view?usp=drive_link)

STEP4-2. Move pre-trained weights into folder model_weights

STEP4-3. Enter following command

```
CUDA_VISIBLE_DEVICES=[GPU Number] python3 IS2D_main.py --num_workers 4 --data_path dataset/BioMedicalDataset --save_path model_weights --train_data_type CVC-ClinicDB --test_data_type CVC-ClinicDB --batch_size 16 --criterion BCE --final_epoch 200 --optimizer_name Adam --lr 0.0001 --LRS_name CALRS --model_name BGANet
```

## For training model, 

STEP4-1. Enter following command

```
CUDA_VISIBLE_DEVICES=[GPU Number] python3 IS2D_main.py --num_workers 4 --data_path dataset/BioMedicalDataset --save_path model_weights --train_data_type CVC-ClinicDB --test_data_type CVC-ClinicDB --batch_size 16 --criterion BCE --final_epoch 200 --optimizer_name Adam --lr 0.0001 --LRS_name CALRS --model_name BGANet --train
```
