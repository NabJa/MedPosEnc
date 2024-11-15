# PosEnc

## What this repo shows
- **Impact of Positional Encodings (PEs) in Medical Imaging:** The study systematically investigates how PEs affect the performance and interpretability of attention-based models across diverse medical imaging tasks, addressing the unique spatial challenges inherent in medical datasets.
- **Comprehensive Task Coverage:** Experiments span classification, regression, segmentation, and image generation using EchoNet-Dynamic, NIH Chest X-ray, and BraTS to capture the complexity of real-world medical scenarios.
- **Advancing Attention-Based Methods:** Findings highlight the critical role of PEs and the superiority of Fourier Feature PE over the commenly used Sinusoidal PE or learned PE.

## Abstract
_This paper presents a crucial exploration into the impact of positional encodings (PEs) on attention-based methods across various medical imaging tasks. While attention mechanisms have demonstrated state-of-the-art performance in diverse benchmarks, their specific influence in the medical domain remains inadequately evaluated. Our experiments contribute to bridging this gap by providing **a nuanced understanding of the role of PEs in medical imaging applications**. The importance of this study lies in the unique challenges posed by medical datasets, where spatial relationships and context are paramount. By conducting comprehensive experiments on classification, segmentation, object detection, image generation, and self-supervised learning tasks, we aim to elucidate the significance of PEs in enhancing the interpretability and performance of attention-based models. The datasets selected for experimentation, including EchoNet-Dynamic, NIH Chest X-ray, and BraTS, represent diverse modalities, medical conditions and dimensionalities, reflecting the complexity of real-world medical scenarios.  This research contributes to advancing the application of attention-based methods in medical imaging by systematically assessing the impact of positional encodings. The findings not only fill a critical void in the literature but also provide practitioners and researchers in the medical field with valuable insights for designing more effective and interpretable deep learning models tailored to medical imaging challenges._

## Datasets
 
- BraTS:            https://www.synapse.org/#!Synapse:syn51156910/wiki/
- EchoNet-Dynamic:  https://echonet.github.io/dynamic/index.html
- NIH Chest X-ray:  https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest


## Reproducibility
To run any of the experiments run [train.py](./posenc/train.py). All used model architectures can be found in [models.py](./posenc/nets/models.py). The training logic for all tasks are defined in the [modules](./posenc/modules) folder.  
