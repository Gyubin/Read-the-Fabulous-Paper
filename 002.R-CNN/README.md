# 002. R-CNN

https://arxiv.org/abs/1311.2524

by Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik

## 1. 요약

Not yet.

## 2. 전문 번역

### 0. Abstract

Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012—achieving a mAP of 53.3%.

> PASCAL VOC 데이터셋으로 평가되는 Object detection 성능은 최근 수년간 정체되어있었다. 가장 성능이 좋은 방법은 복잡한 앙상블 시스템이다. 전형적으로 high-level context의 여러 low-level 이미지 피처를 결합해서 만들었다. 이 논문에서 우리는 단순하고 scalable한 detection 알고리즘을 제안하고, 이는 이전의 최고 성능 점수에 비해 mAP를 30% 올렸다.

Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at http://www.cs.berkeley.edu/ ̃rbg/rcnn.

> 우리의 접근은 2가지 중요한 인사이트를 통해서였다. 첫 째는 CNN을 region proposal에 적용했다는 것, 둘 째는 labeled 데이터가 적기 떄문에 다른 보조적이고 쉬운 task에서 pretraining된 weight를 쓰고, 도메인에 맞게 fine tuning 하는 것이다. 그러면 좋은 결과가 나온다. 우리가 region proposal과 CNN을 결합했기 때문에 이 방법을 Region with CNN features 라는 의미로 R-CNN이라 부른다. 비슷한 CNN 아키텍쳐에서 sliding window 개념을 사용한 OverFeat과 R-CNN을 비교했고 R-CNN이 훨씬 좋음을 알아냈다.

### 1. Introduction

Features matter. The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT [29] and HOG [7]. But if we look at performance on the canonical visual recognition task, PASCAL VOC object detection [15], it is generally acknowledged that progress has been slow during 2010-2012, with small gains obtained by building ensemble systems and employing minor variants of successful methods.

> Feature가 중요하다. 지난 수십년간의 visual recognition task는 SIFT와 HOG를 많이 사용했다. 하지만 PASCAL VOC 대회에서의 performance를 본다면 2010-2012년 사이의 발전이 매우 느렸음을 알 수 있다. 앙상블 시스템이나 다른 작은 변화를 이용해서 적은 발전만 이룩했었다.

... skip past works ...

CNNs saw heavy use in the 1990s (e.g., [27]), but then fell out of fashion with the rise of support vector machines. In 2012, Krizhevsky et al. [25] rekindled interest in CNNs by showing substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9, 10]. Their success resulted from training a large CNN on 1.2 million labeled images, together with a few twists on LeCun’s CNN (e.g., max(x, 0) rectify- ing non-linearities and “dropout” regularization).

> CNN은 1990년대에 자주 쓰여졌지만 SVM이 뜨면서 잘 쓰여지지 않게됐다. 2012년에 와서 Imagenet challenge에서 큰 성능 진보를 만들어낸 AlexNet이 CNN에 대한 관심을 재점화했다. AlexNet의 성공은 120만개의 labeled 이미지에 대한 큰 CNN을 학습한 것과 ReLU, dropout 기법을 사용한 것에 기인한다.

The significance of the ImageNet result was vigorously debated during the ILSVRC 2012 workshop. The central issue can be distilled to the following: To what extent do the CNN classification results on ImageNet generalize to object detection results on the PASCAL VOC Challenge?

> 2012 ILSVRC 워크샵에서 AlexNet의 발전은 활발한 토론을 이끌어냈다. 가장 중심 이슈는 classification 결과의 성능 향상을 어떻게 PASCAL VOC 데이터셋의 object detection task로 일반화시킬 것인가였다.

We answer this question by bridging the gap between image classification and object detection. This paper is the first to show that a CNN can lead to dramatically higher ob- ject detection performance on PASCAL VOC as compared to systems based on simpler HOG-like features. To achieve this result, we focused on two problems: localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data.

> 우리는 이 문제에 대한 답을 classification과 object detection의 차이를 잇는 것으로 냈다. 이 논문은 CNN이 object detection task에서 HOG같은 feature를 활용하는 것보다 더 나은 성능을 내는 것을 보여준 첫 논문이다. 결과를 얻기 위해 2가지 문제에 집중했다. localizing object에 Deep network를 어떻게 사용할 것인가와 capacity가 높은 모델을 적은 양의 detection 데이터로만 학습해야 하는 점이다.

Unlike image classification, detection requires localizing (likely many) objects within an image. One approach frames localization as a regression problem. However, work from Szegedy et al. [38], concurrent with our own, indi- cates that this strategy may not fare well in practice (they report a mAP of 30.5% on VOC 2007 compared to the 58.5% achieved by our method).

> image classification과 달리 detection은 이미지 내에서 여러 object를 localizing하는 것이 필요하다. 첫 번째 접근은 localization을 regression 문제로 푸는 것이다. R-CNN과 같이 나온 Szegedy의 작업은 이러한 방식이 좋지 않음을 보여준다. R-CNN의 58.5%와 비교해서 30.5%의 낮은 mAP만 보여주었다.

An alternative is to build a sliding-window detector. CNNs have been used in this way for at least two decades, typically on constrained object cat- egories, such as faces [32, 40] and pedestrians [35]. In order to maintain high spatial resolution, these CNNs typically only have two convolutional and pooling layers. We also considered adopting a sliding-window approach. However, units high up in our network, which has five convolutional layers, have very large receptive fields (195 × 195 pixels) and strides (32×32 pixels) in the input image, which makes precise localization within the sliding-window paradigm an open technical challenge.

> 또 다른 대안은 sliding-window detector를 사용하는 것이다. CNN은 이런 방식으로 20년간 쓰여져왔는데 주로 detecting하는 물체가 얼굴이나 보행자로 제한된 조건이었다. 높은 spatial resolution을 유지하기 위해 이러한 CNN은 오직 2개의 convolution과 pooling layer만 사용했다. 우리는 sliding-window 방식 역시 사용하려고 고려했지만 unit이 네트워크에서 많아졌고, 5개의 conv layer와 input image에서 매우 큰 receptive fields(195x195)와 strides(32x32)를 가지게 되었다. 이는 sliding-window 패러다임에서 정확한 localization을 하도록 이끈다.
