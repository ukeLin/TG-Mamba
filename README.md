# TG-Mamba

TG-Mamba: Leveraging text guidance for predicting tumor mutation burden in lung cancer

Published in: Computerized Medical Imaging and Graphics

## Project Structure

```
TG-Mamba/
├── MedMamba.py           # Implementation of medical Mamba model
├── TextAttention.py      # Text-guided attention mechanism
├── vmamba.py            # Vision Mamba implementation
├── mil_head.py          # Multiple instance learning head
├── my_dataset.py        # Dataset loading and preprocessing
├── train1.py            # Training script phase 1
├── train2.py            # Training script phase 2
├── eval.py              # Evaluation script
└── svsCut/              # WSI processing utilities
    ├── run.py           # WSI patch extraction script
    └── utils/           # Utility functions for WSI processing
```

## Usage

1. **Data Preparation**
   ```bash
   # Process WSI files and extract patches
   python svsCut/run.py --input_dir path/to/wsi --output_dir path/to/patches
   ```

2. **Training**
   ```bash
   # Phase 1: Initial model training
   python train1.py --data_path path/to/data --text_path path/to/text --batch-size 64 --lr 5e-4 --epochs 100

   # Phase 2: Comparative training with baseline models
   python train2.py --data_path path/to/data --text_path path/to/text --batch-size 64 --lr 5e-5 --epochs 100
   ```

3. **Evaluation**
   ```bash
   python eval.py --model_path path/to/model --test_data path/to/test
   ```

## Training Strategy

### Phase 1 (train1.py)
- Initial training of TG-Mamba model
- Higher learning rate (5e-4) and batch size (64)
- Extended training epochs (100) for thorough model optimization

### Phase 2 (train2.py)
- Comparative experiments with baseline models
- Lower learning rate (5e-5) and batch size (32)
- Early stopping mechanism (patience=10)
- Comprehensive evaluation metrics
- Support for multiple baseline models including:
  - Traditional CNNs (VGG19, ResNet, DenseNet)
  - Vision Transformers (Swin)
  - Modern architectures (ConvNeXt, CellViT, MedViT)

## Acknowledgments

Our code is based on [MedMamba](https://github.com/YubiaoYue/MedMamba). We thank the authors for their excellent work in medical image classification.

## Citation

```bibtex
@article{yu2025tg,
  title={TG-Mamba: Leveraging text guidance for predicting tumor mutation burden in lung cancer},
  author={Yu, Chunlin and Meng, Xiangfu and Li, Yinhao and Zhao, Zheng and Zhang, Yongqin},
  journal={Computerized Medical Imaging and Graphics},
  pages={102626},
  year={2025},
  publisher={Elsevier}
}
```

