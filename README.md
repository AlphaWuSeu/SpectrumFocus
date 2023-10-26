# A Plug-and-play Spectrum Focus Module in Vision Transformers

**Paper Title**: A Plug-and-play Spectrum Focus Module in Vision Transformers

**Abstract**: 
The Vision Transformer (ViT) excels at discerning semantic relationships in images but often overlooks intricate local features unless extensively trained. Drawing parallels from human visual recognition, which is sensitive to shape, texture, and color, this research highlights similar patterns in machine recognition. However, existing methodologies introduce additional structures to capture local texture and shape, and an overemphasis risks obfuscating critical distinctions. We propose the “Spectrum Focus Module” utilizing wavelet decomposition to differentiate texture and shape while safeguarding color. This module enhances feature extraction, balancing performance across texture, shape, and color without redundant computations. Empirical results demonstrate its adaptability across multiple ViT derivatives, amplifying recognition capabilities on varied texture datasets.


## How to Use
1. **Preparation**: Before running the code, make sure you've prepared the necessary datasets and modified the configuration files accordingly.
2. **Training**:
python train.py 


## Model Architecture
Our model builds upon the foundation of SwinViT but introduces the Spectrum Focus Module to enhance its ability to recognize textures. The module is a plug-and-play solution that can be added to any layer of the Vision Transformer.

## Dataset
1. **ImageNet100**: https://www.kaggle.com/datasets/ambityga/imagenet100
2. **feature-bias_dataset**: https://github.com/gyhandy/Humanoid-Vision-Engine
4. **KTH-TIP**: https://www.csc.kth.se/cvap/databases/kth-tips/index.html
5. **FMD**: https://people.csail.mit.edu/lavanya/fmd.html
6. **4D-Light**: http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV16/LF_dataset.zip
7. **DTD**: http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV16/LF_dataset.zip
8. **CUReT**: https://www.cs.columbia.edu/CAVE/software/curet/
9. **GTOs**: https://onedrive.live.com/?authkey=%21ALmEnEv4R5LyxT4&cid=E7E6D87809E2DF64&id=E7E6D87809E2DF64%21112&parId=E7E6D87809E2DF64%21108&action=defaultclick
10. **MIT-Indoor**: http://web.mit.edu/torralba/www/indoor.html
11. **MINC**: http://opensurfaces.cs.cornell.edu/publications/minc/#download
12. **GTOs_M**: https://drive.google.com/file/d/1Hd1G7aKhsPPMbNrk4zHNJAzoXvUzWJ9M/view


## Results
Our method shows superior performance on various texture datasets, outperforming standard Vision Transformers. Detailed results can be found in our paper.
| Name    | Category | Size    |Swin-Ti  | SFM_Swin-Ti |
|---------|----------|---------|---------|-------------|
| KTH     | 10       | 0.8k    | 99.26   | 99.47       |
| FMD     | 10       | 1k      | 42.51   | 47.70       |
| 4Dlight | 12       | 1.2k    | 58.67   | 59.67       |
| KTH-2b  | 10       | 4.3k    | 99.59   | 99.87       |
| DTD     | 47       | 5.6k    | 58.53   | 61.69       |
| CuReT   | 61       | 9.8k    | 99.52   | 99.66       |
| GTOs(d) | 39       | 32.7k   | 66.27   | 69.15       |
| GTOs(c) | 39       | 34.5k   | 79.33   | 80.50       |
| MIT     | 67       | 40.7k   | 67.19   | 69.88       |
| MINC    | 23       | 57.5k   | 74.50   | 77.00       |
| GTOs-M  | 65       | 100k    | 88.92   | 89.63       |


## Acknowledgments
This work was inspired and built upon the original [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) by Microsoft. We thank the authors for their contribution.

## Citing our work
If you find our work useful or use our code in your research, please consider citing our paper:
