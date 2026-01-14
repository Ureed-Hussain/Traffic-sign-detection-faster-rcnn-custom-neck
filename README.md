# Traffic sign detection using Faster-RCNN Custom Neck

This repository extends my baseline project Traffic Sign Detection using Faster R-CNN by introducing a custom neck architecture designed to enhance multi-scale feature representation while keeping the backbone and detection head unchanged.
The baseline repository, which establishes a clean and reproducible Faster R-CNN detector using a ResNet50 + FPN backbone on the German Traffic Sign Detection Benchmark (GTSDB), can be found here:

👉 Baseline Repository:
[https://github.com/Ureed-Hussain/traffic-sign-detection-faster-rcnn/tree/main](https://github.com/Ureed-Hussain/traffic-sign-detection-faster-rcnn)

In this project, the primary goal is architectural isolation: to study how modifying only the neck affects detection performance, convergence behavior, and feature quality, without confounding factors from backbone or head changes. This makes the comparison with the baseline model clean, fair, and interpretable.

The overall workflow and model variants used in the project are illustrated in the accompanying flowchart.

<p align="center">
<img width="500" alt="Detection_flowchart" src="https://github.com/user-attachments/assets/5ad30c66-b45b-4cda-b254-67424d1e85f1" />
</p>

## Repository Structure

* **custom_neck_detector.ipynb**: Implements Faster R-CNN with a ResNet50 backbone and a custom StrongNeck-enhanced FPN.

## Dataset Description, German Traffic Sign Detection Benchmark (GTSDB)

The GTSDB dataset is part of the IJCNN 2013 Traffic Sign Detection benchmark and contains real-world traffic scene images with annotated traffic signs. The dataset provides high-resolution images and standardized ground-truth annotations for evaluating object detection models.

###  Image Format

* Images are stored in PPM format with a resolution of 1360 × 800 pixels.

* Each image contains 0–6 traffic signs, appearing at sizes between 16×16 and 128×128 pixels.

* Signs may vary in perspective, lighting, and environment, making the dataset suitable for training robust detectors.

### Annotation Format

Annotations are provided in a semicolon-separated CSV file (gt.txt), where each entry contains:

* Filename

* Bounding box: x1; y1; x2; y2

* Class ID: integer representing the traffic sign category

#### Example fields:

image_xx.ppm; left; top; right; bottom; class_id

The dataset follows the class ID definitions described in the official ReadMe.txt file of the GTSDB package.


### Dataset Splits

The GTSDB dataset includes the following official splits (IJCNN 2013):

* FullIJCNN2013.zip → 900 total images

* TrainIJCNN2013.zip → 600 training images

* TestIJCNN2013.zip → 300 test images (no ground truth)

* gt.txt → ground-truth annotations for training and evaluation

In the project, after cleaning and filtering bounding boxes, I load:

* Total samples used: 506 images

* Train split: 404 images

* Validation split: 102 images

### Dataset Download

The dataset is downloaded automatically when running the data upload section in the notebooks. Users only need to upload their kaggle.json file, after which the code handles authentication, dataset download, and extraction.

### Data Visualization
I visualize samples from the dataset with bounding boxes and class IDs overlaid on the images.
Each numeric label corresponds to a traffic sign class defined in the GTSDB label specification.

<p align="center">
<img width="794" height="498" alt="image" src="https://github.com/user-attachments/assets/7d012d07-54af-4780-8517-7cd8e31cc608" />
</p>

### Data Augmentation
To improve robustness, I apply lightweight augmentations using Albumentations, ensuring consistency between images and bounding boxes:

~~~python
train_aug = A.Compose([
    A.Rotate(limit=8, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
~~~

These augmentations help improve model robustness by introducing small rotations and brightness/contrast variations.

<p align="center">
<img width="950" height="290" alt="image-1" src="https://github.com/user-attachments/assets/982e42bc-1f4d-426a-9130-194b3301dbd6" />
</p>


### Train/Validation Split
I split the dataset into 80% training and 20% validation using a reproducible train_test_split based on image indices. Both subsets share the same annotations, with the training set using augmentations and the validation set kept clean for unbiased evaluation.
~~~
Loaded 506 images with bounding boxes. (transforms=no)
Total samples: 506 | Train: 404 | Val: 102
Loaded 404 images with bounding boxes. (transforms=yes)
Loaded 102 images with bounding boxes. (transforms=no)
~~~

## Model Description
This detector follows the canonical Faster R-CNN design, composed of three major components:

1. Backbone: ResNet50
2. Neck: Custom StrongNeck-enhanced FPN (proposed in this work)
3. Head: Standard Faster R-CNN detection head

Only the neck is modified; the backbone and head remain exactly the same as in the baseline implementation.

<p align="center">
<img width="865" height="540" alt="strong_neck_diagram" src="https://github.com/user-attachments/assets/f8fb0ea8-9c03-4432-9c21-1668199b84e6" />
</p>

###  Backbone: ResNet50
The backbone is identical to the baseline Faster R-CNN model. A ResNet50 network initialized with ImageNet-pretrained weights is used for feature extraction. Pretraining stabilizes optimization and improves generalization, especially given the limited size of the GTSDB dataset.

The backbone produces multi-scale feature maps (C2–C5), which serve as the input to the neck module. Since this experiment focuses exclusively on neck design, the backbone is intentionally left unchanged to act as a consistent and reliable feature generator.



### Custom Neck Architecture

The central contribution of this repository is the Custom StrongNeck, which replaces the standard FPN refinement path. Instead of directly passing FPN features to the detection head, each pyramid level (P2–P5) is processed by a StrongNeck block that enhances multi-scale context, spatial focus, and channel discrimination.

#### StrongNeck Design

Each StrongNeck block consists of three complementary components:

##### 1. Multi-Dilated Convolution Branches
Three parallel 3×3 convolutions with dilation rates 1, 2, and 3 are applied to capture local details and broader contextual information simultaneously. These outputs are fused via a 1×1 convolution, enabling effective receptive field expansion without excessive computational cost.

##### 2. Depthwise–Pointwise Bottleneck
A depthwise convolution refines spatial structures independently across channels, followed by a pointwise (1×1) convolution to mix channel information. This bottleneck improves representational efficiency while preserving spatial sensitivity—important for detecting small traffic signs.

#### 3. Channel & Spatial Attention (CBAM-style)
To further refine features, an attention mechanism is applied:

* Channel Attention:
Learns which feature channels are most informative using both average and max pooling followed by a lightweight MLP.

* Spatial Attention:
Emphasizes relevant spatial regions using pooled spatial descriptors.

A residual connection adds the attention-refined output back to the original input, improving gradient flow and stabilizing training.

Finally, a wrapper module (CustomStrongFPN) applies StrongNeck blocks consistently across all pyramid levels before forwarding the enhanced feature maps to the Faster R-CNN head.

### Head: Faster R-CNN
The detection head remains the standard Faster R-CNN head provided by TorchVision, including:

* Region Proposal Network (RPN)
* RoIAlign
* Classification layers
* Bounding box regression layers

The only difference is the input feature quality. Instead of receiving raw FPN outputs, the head now operates on StrongNeck-enhanced feature maps with richer multi-scale context and stronger spatial and channel focus.

Because both the backbone and head are unchanged, any observed performance differences relative to the baseline can be directly attributed to the custom neck design.

## Training Configuration

Training settings are kept identical to the baseline project for consistency:

* Epochs: 25
* Mixed Precision Training (AMP): Enabled
* Optimizer and scheduler: Same as baseline
* Checkpointing: Best validation model saved
* Logging: Batch-wise loss monitoring

This controlled setup ensures that architectural effects are not confounded by training hyperparameters.

### Training and Validation Loss

The custom-neck model shows slightly slower initial convergence, with training loss starting at 0.42, compared to the baseline. However, as training progresses, the model adapts effectively and reaches a final validation loss of 0.12 by epoch 25.

This behavior suggests that the stronger feature refinement introduced by the custom neck requires additional adaptation early in training, but ultimately provides competitive—and more expressive—feature representations for the detection task.

![Custom_neck_curve](https://github.com/user-attachments/assets/46e201ac-c9b4-4843-81ff-f8939ead3e62)


## Results

| Metric | Value |
|------|------|
| Precision | 90.0% |
| Recall | 90.0% |
| mAP | 90.0% |

### Qualitative Result
The following visualization shows predicted bounding boxes closely matching the ground truth, demonstrating strong localization and classification performance.

<img width="1016" height="1490" alt="Custom_Neck_Image" src="https://github.com/user-attachments/assets/1038f0a9-2398-4511-bc1a-1df9e7225f06" />


## Conclusion

In this repository, I extend a well-established Faster R-CNN baseline by introducing a custom StrongNeck architecture while keeping the backbone and detection head unchanged. This design choice enables a clean and interpretable analysis of how enhanced feature aggregation, attention mechanisms, and receptive field expansion affect traffic sign detection performance.

The results demonstrate that modifying only the neck can meaningfully influence convergence behavior and final performance, validating the importance of neck design in two-stage detectors. This repository serves as an intermediate step between a standard pretrained detector and a fully custom Faster R-CNN implementation, bridging correctness-focused baselines and deeper architectural experimentation.

## Next Steps and Related Repositories

This repository represents the completion of the custom neck stage in my structured exploration of traffic sign detection using Faster R-CNN. With the backbone, training pipeline, and detection head kept fixed, this project isolates and evaluates the impact of a redesigned neck architecture on multi-scale feature representation and detection performance.

The next and final step in this research trajectory is a full from-scratch implementation of Faster R-CNN, where all architectural components are explicitly designed and implemented without relying on high-level TorchVision abstractions.

## Faster R-CNN Implemented from Scratch

Repository: faster-rcnn-from-scratch-traffic-sign-detection

In this upcoming repository, I implement Faster R-CNN entirely from the ground up, including the backbone, neck, Region Proposal Network (RPN), RoIAlign, classification head, and bounding box regression head. Rather than focusing on pretrained convenience, this implementation emphasizes architectural transparency, modularity, and a deeper understanding of the internal mechanics of two-stage object detectors.

Together, the three repositories—baseline Faster R-CNN, Faster R-CNN with a custom neck, and Faster R-CNN implemented from scratch—form a progressive and well-scoped study of traffic sign detection, advancing from a stable reference model to targeted architectural customization, and finally to a complete end-to-end reimplementation.
