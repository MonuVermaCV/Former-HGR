# Former-HGR: Hand Gesture Recognition With Hybrid Feature-Aware Transformer


Monu Verma, Garvit Gopalani, Saiyam Bharara, Santosh Kumar Vipparthi, Subrahmanyam Murala and Mohamed Abdel-Mottaleb. "Former-HGR: Hand Gesture Recognition with Hybrid Feature-Aware Transformer. doi: https://doi.org/10.1109/LSENS.2025.3566022

# Setup
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

# Training on OUHANDS
Step 1: download dataset and place to folder "data".

Step 2: fill in all the your_dataset_path in dataloader/dataset_OUHANDS.py

Step 3: download ir50.pth https://drive.google.com/file/d/1rgxthxEYUKUj6Y_P22_1Dika8yR10Mb8/view?usp=drive_link and place in models/

Step 3: run python main_OUHANS.py
 # Citation
 If you find our work useful, please consider citing our paper:
 
 @ARTICLE{11005620,
  author={Verma, Monu and Gopalani, Garvit and Bharara, Saiyam and Vipparthi, Santosh Kumar and Murala, Subrahmanyam and Abdel-Mottaleb, Mohamed},
  journal={IEEE Sensors Letters}, 
  title={Former-HGR: Hand Gesture Recognition With Hybrid Feature-Aware Transformer}, 
  year={2025},
  volume={},
  number={},
  pages={1-4},
  keywords={Hands;Transformers;Feature extraction;Gesture recognition;Training;Convolutional neural networks;Convolution;Data mining;Accuracy;Tensors;Hand gesture recognition using imaging/Optical sensors;hybrid feature-aware transformer;transformer},
  doi={10.1109/LSENS.2025.3566022}}
