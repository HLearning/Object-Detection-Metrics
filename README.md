

<p align="left">
    <a href="https://zenodo.org/badge/latestdoi/134606465">
        <img src="https://zenodo.org/badge/134606465.svg"/></a>
    <a href="https://opensource.org/licenses/MIT" >
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" /></a>
    <a href="https://github.com/rafaelpadilla/Object-Detection-Metrics/raw/master/paper_survey_on_performance_metrics_for_object_detection_algorithms.pdf">
        <img src="https://img.shields.io/badge/paper-published-red"/></a>
</p>

## 翻译名词说明
- Metrics： 评估指标
- ground truth： 真实
- Precision： 查准率， 准确率
- Recall： 查全率， 召回率
- True Positive：真正例， 真阳性
- False Positive：假正例， 假阳性
- False Negative：假反例， 假阴性
- True Negative：真反例， 真阴性
- Intersection Over Union： 交并比
- object detection： 目标检测， 对象检测
- Precision x Recall curve： 查准率-查全率曲线， PR曲线


## 引用
此项工作被IWSSIP 2020接收并发表。如果您使用此代码进行研究，请考虑引用：
```
@INPROCEEDINGS {padillaCITE2020,
    author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},
    title     = {A Survey on Performance Metrics for Object-Detection Algorithms}, 
    booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)}, 
    year      = {2020},
    pages     = {237-242},}
```
[这里](https://github.com/rafaelpadilla/Object-Detection-Metrics/raw/master/paper_survey_on_performance_metrics_for_object_detection_algorithms.pdf)下载论文


# 目标检测评估指标
这个项目的初心是因为在不同的工作和实现中对目标检测的评估指标缺乏共识。
尽管在线比赛使用自己的评估标准来评估对象检测的任务，
但只有其中一些提供参考代码段来计算检测到的对象的准确性。

研究人员想要使用与竞赛提供的数据集不同的数据集来评估他们的工作，需要实现他们自己版本的评估指标。 
有时，错误或不同的实现可能会产生不同且有偏差的结果。 
理想情况下，为了在不同的方法之间进行可靠的基准测试，有必要有一个灵活的实现，
无论使用什么数据集，任何人都可以使用。

该项目提供简单易用的功能，实现和最流行的对象检测竞赛所使用的相同评估指标。 
我们的实现不需要将您的检测模型修改为复杂的输入格式，从而避免了转换为XML或JSON文件。 
我们简化了输入数据（真相边界框和检测到的边界框），并将学术界和挑战所使用的主要指标集成到一个项目中，
我们将我们的实施方案与官方实施方案进行了仔细比较，结果完全相同。 

## 目录

- [初心](#metrics-for-object-detection)
- [不同的比赛，不同的评估指标](#different-competitions-different-metrics)
- [重要的定义](#important-definitions)
- [评估指标](#metrics)
  - [查准率-查全率曲线](#precision-x-recall-curve)
  - [平均查准率](#average-precision)
    - [11点插值](#11-point-interpolation)
    - [所有点插值](#interpolating-all-points)
- [**如何使用此项目**](#how-to-use-this-project)
- [参考](#references)



<a name="different-competitions-different-metrics"></a> 
## 不同的比赛，不同的评估指标 

* **[PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)** 提供了一个Matlab脚本来评估检测到的对象的质量. 
参赛者可以在提交结果之前使用提供的Matlab脚本来测量其检测的准确性。 可以在[此处](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000)访问解释目标检测评估指标的官方文档。 
当前PASCAL VOC目标检测挑战赛所使用的评估指标是：**查准率-查全率曲线**和**平均查准率**
PASCAL VOC Matlab评估指标代码从XML文件的读取真实边框，如果要将其应用于其他数据集或特殊情况，则需要对代码进行更改。 
即使如[Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn)之类的项目实现了PASCAL VOC的评估指标，也需要将检测到的边框转换为其特定格式。[Tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md)框架也具有PASCAL VOC评估指标的实现。

* **[COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)** 使用不同的指标来评估不同目标检测算法的准确性。
[在这里](http://cocodataset.org/#detection-eval) , 您可以找到说明12种评估指标的文档，这些评估指标用于评估目标检测在COCO数据集上的性能。 
这项竞赛提供了Python和Matlab代码，因此用户可以在提交结果之前验证自己的分数。 也需要将结果转换为比赛所需的[格式](http://cocodataset.org/#format-results) 

* **[Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)** 使用500个类的平均的平均查准率（mAP）来评估目标检测任务. 

* **[ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)** 考虑每个图像中的每个类的真实框和检测框的重叠区域，为图片定义一个误差值。总误差为计算所有测试数据集图像中最小误差的平均值。[这里](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation) 有更多关于他们评估方法的细节。

## 重要定义  

### 交并比 (IOU)

交并比 (IOU) 是基于Jaccard系数来评估两个边框之间的重叠。他需要一个真实边框 ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) 和一个检测边框![](http://latex.codecogs.com/gif.latex?B_p). 
通过IOU我们可以判断检测是有效的(真正例) 还是无效的(假正例).  
IOU由真实边框和检测边框之间的交集的面积除以他们之间并集的面积得到：

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BIOU%7D%3D%5Cfrac%7B%5Ctext%7Barea%7D%5Cleft%28B_%7Bp%7D%20%5Ccap%20B_%7Bgt%7D%20%5Cright%29%7D%7B%5Ctext%7Barea%7D%5Cleft%28B_%7Bp%7D%20%5Ccup%20B_%7Bgt%7D%20%5Cright%29%7D">
</p>

<!---
\text{IOU}=\frac{\text{area}\left(B_{p} \cap B_{gt} \right)}{\text{area}\left(B_{p} \cup B_{gt} \right)} 
--->

这个图片展示了真实边框（蓝色）和检测边框（红色）之间的IOU。

<!--- IOU --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/iou.png" align="center"/></p>

### 真正例, 假正例, 假反例， 真反例  

使用评估指标过程中的一些基本概念:  
* **真正例 (TP)**: 一次正确的检测。 检测的IOU ≥ _阈值_  
* **假正例 (FP)**: 一次错误的检测。 检测的IOU < _阈值_  
* **假反例 (FN)**: 一个的真实值未检测到。  
* **真反例 (TN)**: 不适用。 它表示一个误判结果。 在目标检测任务中，存在许多不应该在图像内检测到的可能的边界框。 因此，TN是正确但未检测到的所有可能的边界框(图像中有很多的可能框)。 这就是指标不使用它的原因。
_阈值_: 取决于评估指标, 通常取值：50%, 75% 或 95%.

### 查准率

查准率是模型仅识别相关对象的能力。 它是正确的正面预测的百分比，由以下公式给出：

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFP%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

<!---
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}=\frac{\text{TP}}{\text{all detections}}
--->

### 查全率 

查全率是一个模型来找到所有真实边框的能力。这是所有相关的真实边框检测的真正例的百分比为:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFN%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>
<!--- 
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}=\frac{\text{TP}}{\text{all ground truths}}
--->

## 评估指标
在下面的主题中，有一些有关用于目标检测的最受欢迎的评估指标。


### 查准率-查全率 曲线

查准率-查全率曲线是评估目标检测器性能的好方法，因为置信度会通过每个目标类别的绘制曲线来更改。 如果特定类别的目标检测器在查全率增加时保持较高的查准率，则被认为是好的，这意味着如果您改变置信度阈值，则查准率和查全率仍然很高。 判定一个目标检测器好坏的另一种方式是寻找只能识别一种相关物体的检测器(0假正例=高查准率)，找出所有真实目标(0 假反例 = 高查全率)。

不良的目标检测器需要增加检测到的对象的数量(增加假阳性=降低精度)，以便检索所有真实对象(高查全率)。 这就是为什么查准率-查全率曲线通常从高查准率开始，随着查全率的增加而降低。 您可以在下一个主题(平均查准率)中查看查准率-查全率曲线的示例。 这种曲线被PASCAL VOC 2012挑战赛使用，并在我们的实现中可用。

### 平均查准率 

另一种比较目标检测器性能的方法是计算查准率-查全率曲线的曲线下面积（AUC）。 由于AP曲线通常是上下弯曲的锯齿形曲线，因此比较同一图中的不同曲线（不同的检测器）通常不是一件容易的事， 因为这些曲线往往会频繁交叉。 这就是为什么平均精度(AP)这一数字指标也可以帮助我们比较不同的检测器。 实际上，AP是在0到1之间的所有查全率上平均的精度。

从2010年开始，PASCAL VOC挑战赛计算AP的方法发生了变化. 目前, **PASCAL VOC挑战执行的插值使用所有数据点，而不是插值只有11个等间隔的点，如他们所述[论文](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)**. 因为我们希望重现它们的默认实现，所以我们的默认代码(如下文所示)遵循它们的最新应用程序(插值所有数据点)。 但是，我们也提供了11点插值方法。

#### 11-point interpolation

The 11-point interpolation tries to summarize the shape of the Precision x Recall curve by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BAP%7D%3D%5Cfrac%7B1%7D%7B11%7D%20%5Csum_%7Br%5Cin%20%5Cleft%20%5C%7B%200%2C%200.1%2C%20...%2C1%20%5Cright%20%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28%20r%20%5Cright%20%29%7D">
</p>
<!---
\text{AP}=\frac{1}{11} \sum_{r\in \left \{ 0, 0.1, ...,1 \right \}}\rho_{\text{interp}\left ( r \right )}
--->

with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cgeq%20r%7D%20%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>
<!--- 
\rho_{\text{interp}} = \max_{\tilde{r}:\tilde{r} \geq r} \rho\left ( \tilde{r} \right )
--->

where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision only at the 11 levels ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r)**.

#### Interpolating all points

Instead of interpolating only in the 11 equally spaced points, you could interpolate through all points <img src="https://latex.codecogs.com/gif.latex?n"> in such way that:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bn%3D0%7D%20%5Cleft%20%28%20r_%7Bn&plus;1%7D%20-%20r_%7Bn%7D%20%5Cright%20%29%20%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29">
</p>
<!---
\sum_{n=0} \left ( r_{n+1} - r_{n} \right ) \rho_{\text{interp}}\left ( r_{n+1} \right )
--->
 
with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cge%20r_%7Bn&plus;1%7D%7D%20%5Crho%20%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>

<!---
\rho_{\text{interp}}\left ( r_{n+1} \right ) = \max_{\tilde{r}:\tilde{r} \ge r_{n+1}} \rho \left ( \tilde{r} \right )
--->


where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

In this case, instead of using the precision observed at only few points, the AP is now obtained by interpolating the precision at **each level**, ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater or equal than ![](http://latex.codecogs.com/gif.latex?r&plus;1)**. This way we calculate the estimated area under the curve.

To make things more clear, we provided an example comparing both interpolations.


#### An ilustrated example 

An example helps us understand better the concept of the interpolated average precision. Consider the detections below:
  
<!--- Image samples 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/samples_1_v2.png" align="center"/></p>
  
There are 7 images with 15 ground truth objects representented by the green bounding boxes and 24 detected objects represented by the red bounding boxes. Each detected object has a confidence level and is identified by a letter (A,B,...,Y).  

The following table shows the bounding boxes with their corresponding confidences. The last column identifies the detections as TP or FP. In this example a TP is considered if IOU ![](http://latex.codecogs.com/gif.latex?%5Cgeq) 30%, otherwise it is a FP. By looking at the images above we can roughly tell if the detections are TP or FP.

<!--- Table 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_1_v2.png" align="center"/></p>

<!---
| Images | Detections | Confidences | TP or FP |
|:------:|:----------:|:-----------:|:--------:|
| Image 1 | A | 88% | FP |
| Image 1 | B | 70% | TP |
| Image 1 |	C	| 80% | FP |
| Image 2 |	D	| 71% | FP |
| Image 2 |	E	| 54% | TP |
| Image 2 |	F	| 74% | FP |
| Image 3 |	G	| 18% | TP |
| Image 3 |	H	| 67% | FP |
| Image 3 |	I	| 38% | FP |
| Image 3 |	J	| 91% | TP |
| Image 3 |	K	| 44% | FP |
| Image 4 |	L	| 35% | FP |
| Image 4 |	M	| 78% | FP |
| Image 4 |	N	| 45% | FP |
| Image 4 |	O	| 14% | FP |
| Image 5 |	P	| 62% | TP |
| Image 5 |	Q	| 44% | FP |
| Image 5 |	R	| 95% | TP |
| Image 5 |	S	| 23% | FP |
| Image 6 |	T	| 45% | FP |
| Image 6 |	U	| 84% | FP |
| Image 6 |	V	| 43% | FP |
| Image 7 |	X	| 48% | TP |
| Image 7 |	Y	| 95% | FP |
--->

In some images there are more than one detection overlapping a ground truth (Images 2, 3, 4, 5, 6 and 7). For those cases the first detection is considered TP while the others are FP. This rule is applied by the PASCAL VOC 2012 metric: "e.g. 5 detections (TP) of a single object is counted as 1 correct detection and 4 false detections”.

The Precision x Recall curve is plotted by calculating the precision and recall values of the accumulated TP or FP detections. For this, first we need to order the detections by their confidences, then we calculate the precision and recall for each accumulated detection as shown in the table below: 

<!--- Table 2 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_2_v2.png" align="center"/></p>

<!---
| 图片 | 检测结果 | 置信度 |  TP | FP | Acc TP | Acc FP | 查准率 | 查全率 |
|:------:|:----------:|:-----------:|:---:|:--:|:------:|:------:|:---------:|:------:|
| Image 5 |	R	| 95% | 1 | 0 | 1 | 0 | 1       | 0.0666 |
| Image 7 |	Y	| 95% | 0 | 1 | 1 | 1 | 0.5     | 0.6666 |
| Image 3 |	J	| 91% | 1 | 0 | 2 | 1 | 0.6666  | 0.1333 |
| Image 1 | A | 88% | 0 | 1 | 2 | 2 | 0.5     | 0.1333 |
| Image 6 |	U	| 84% | 0 | 1 | 2 | 3 | 0.4     | 0.1333 |
| Image 1 |	C	| 80% | 0 | 1 | 2 | 4 | 0.3333  | 0.1333 |
| Image 4 |	M	| 78% | 0 | 1 | 2 | 5 | 0.2857  | 0.1333 |
| Image 2 |	F	| 74% | 0 | 1 | 2 | 6 | 0.25    | 0.1333 |
| Image 2 |	D	| 71% | 0 | 1 | 2 | 7 | 0.2222  | 0.1333 |
| Image 1 | B | 70% | 1 | 0 | 3 | 7 | 0.3     | 0.2    |
| Image 3 |	H	| 67% | 0 | 1 | 3 | 8 | 0.2727  | 0.2    |
| Image 5 |	P	| 62% | 1 | 0 | 4 | 8 | 0.3333  | 0.2666 |
| Image 2 |	E	| 54% | 1 | 0 | 5 | 8 | 0.3846  | 0.3333 |
| Image 7 |	X	| 48% | 1 | 0 | 6 | 8 | 0.4285  | 0.4    |
| Image 4 |	N	| 45% | 0 | 1 | 6 | 9 | 0.7     | 0.4    |
| Image 6 |	T	| 45% | 0 | 1 | 6 | 10 | 0.375  | 0.4    |
| Image 3 |	K	| 44% | 0 | 1 | 6 | 11 | 0.3529 | 0.4    |
| Image 5 |	Q	| 44% | 0 | 1 | 6 | 12 | 0.3333 | 0.4    |
| Image 6 |	V	| 43% | 0 | 1 | 6 | 13 | 0.3157 | 0.4    |
| Image 3 |	I	| 38% | 0 | 1 | 6 | 14 | 0.3    | 0.4    |
| Image 4 |	L	| 35% | 0 | 1 | 6 | 15 | 0.2857 | 0.4    |
| Image 5 |	S	| 23% | 0 | 1 | 6 | 16 | 0.2727 | 0.4    |
| Image 3 |	G	| 18% | 1 | 0 | 7 | 16 | 0.3043 | 0.4666 |
| Image 4 |	O	| 14% | 0 | 1 | 7 | 17 | 0.2916 | 0.4666 |
--->
 
 Plotting the precision and recall values we have the following *Precision x Recall curve*:
 
 <!--- Precision x Recall graph --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/precision_recall_example_1_v2.png" align="center"/>
</p>
 
As mentioned before, there are two different ways to measure the interpolted average precision: **11-point interpolation** and **interpolating all points**. Below we make a comparisson between them:

#### Calculating the 11-point interpolation

The idea of the 11-point interpolated average precision is to average the precisions at a set of 11 recall levels (0,0.1,...,1). The interpolated precision values are obtained by taking the maximum precision whose recall value is greater than its current recall value as follows: 

<!--- interpolated precision curve --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/11-pointInterpolation.png" align="center"/>
</p>

By applying the 11-point interpolation, we have:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%5Csum_%7Br%5Cin%5C%7B0%2C0.1%2C...%2C1%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%20%5Cleft%20%28%201&plus;0.6666&plus;0.4285&plus;0.4285&plus;0.4285&plus;0&plus;0&plus;0&plus;0&plus;0&plus;0%20%5Cright%20%29)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%2026.84%5C%25)


#### Calculating the interpolation performed in all points

By interpolating all points, the Average Precision (AP) can be interpreted as an approximated AUC of the Precision x Recall curve. The intention is to reduce the impact of the wiggles in the curve. By applying the equations presented before, we can obtain the areas as it will be demostrated here. We could also visually have the interpolated precision points by looking at the recalls starting from the highest (0.4666) to 0 (looking at the plot from right to left) and, as we decrease the recall, we collect the precision values that are the highest as shown in the image below:
    
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision_v2.png" align="center"/>
</p>
  
Looking at the plot above, we can divide the AUC into 4 areas (A1, A2, A3 and A4):
  
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision-AUC_v2.png" align="center"/>
</p>

计算总面积, 我们就得到了 AP:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20A1%20&plus;%20A2%20&plus;%20A3%20&plus;%20A4)  
  
![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bwith%3A%7D)  
![](http://latex.codecogs.com/gif.latex?A1%20%3D%20%280.0666-0%29%5Ctimes1%20%3D%5Cmathbf%7B0.0666%7D)  
![](http://latex.codecogs.com/gif.latex?A2%20%3D%20%280.1333-0.0666%29%5Ctimes0.6666%3D%5Cmathbf%7B0.04446222%7D)  
![](http://latex.codecogs.com/gif.latex?A3%20%3D%20%280.4-0.1333%29%5Ctimes0.4285%20%3D%5Cmathbf%7B0.11428095%7D)  
![](http://latex.codecogs.com/gif.latex?A4%20%3D%20%280.4666-0.4%29%5Ctimes0.3043%20%3D%5Cmathbf%7B0.02026638%7D)  
   
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.0666&plus;0.04446222&plus;0.11428095&plus;0.02026638)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.24560955)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cmathbf%7B24.56%5C%25%7D)  

两种不同插值方法之间的结果略有不同：所有点插值和11点插值分别为24.56％和26.84％。  

我们的默认实现与VOC Pascal相同：所有点插值。 如果要使用11点插值，请更改使用函数的参数```method=MethodAveragePrecision.EveryPointInterpolation``` 到 ```method=MethodAveragePrecision.ElevenPointInterpolation```.   

如果要重现这些结果，请参阅 **[样例2](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2/)**.
<!--In order to evaluate your detections, you just need a simple list of `Detection` objects. A `Detection` object is a very simple class containing the class id, class probability and bounding boxes coordinates of the detected objects. This same structure is used for the groundtruth detections.-->

## 如何使用此项目

创建这个项目是为了以一种非常简单的方式评估您的检测结果. 如果您想用最常用的对象检测评估指标来评估您的算法，那么您就来对了地方。  

[样例1](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_1) 和 [样例2](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2) 是一些实际的例子，演示了如何直接访问这个项目的核心功能，从而在评估的使用上提供了更大的灵活性。但如果您不想花时间理解我们的代码，请参阅下面的说明，以便轻松评估您的检测结果：
按照以下步骤开始评估您的检测结果:

1. [创建你的真实文件](#create-the-ground-truth-files)
2. [创建你的检测文件](#create-your-detection-files)
3. 若是 **Pascal VOC 评估指标**, 运行此命令: `python pascalvoc.py`  
   如果你想复现上面的例子, 运行命令: `python pascalvoc.py -t 0.3`
4. (可选) [您可以使用可选参数来控制IOU阈值或边框格式等等](#optional-arguments)

### 创建真实文件

- 在文件夹 **groundtruths/**中为每一个图片创建一个单独的真实文本.
- 在这些文件中，每行的格式都应该是: `<类别名称> <x1> <y1> <x2> <y2>`.    
- 例如： 图片 "2008_000034.jpg"的真实边框在文本"2008_000034.txt"中表示为:
  ```
  bottle 6 234 45 362
  person 1 156 103 336
  person 36 111 198 416
  person 91 42 338 500
  ```
    
另外，如果您愿意，也可以将边框的格式设置为: `<类别名称> <x1> <y1> <w> <h>` (看这里 [**\***](#asterisk) 如何使用它). 在这种情况下，您的“2008_000034.txt”将表示为:
  ```
  bottle 6 234 39 128
  person 1 156 102 180
  person 36 111 162 305
  person 91 42 247 458
  ```

### 创建检测文件

- 在文件夹 **detections/**中为每一个图片创建一个单独的检测文本.
- 检测边框文本的名称必须要和真实边框的文本名称相对应 (例如： "detections/2008_000182.txt" 对应的真实边框的文本为: "groundtruths/2008_000182.txt").
- 在这些文件中，每行的格式都应该是: `<类别名称> <置信度> <x1> <y1> <x2> <y2>` (看这里 [**\***](#asterisk) 如何使用它). 
- 例如： "2008_000034.txt":
    ```
    bottle 0.14981 80 1 295 500  
    bus 0.12601 36 13 404 316  
    horse 0.12526 430 117 500 307  
    pottedplant 0.14585 212 78 292 118  
    tvmonitor 0.070565 388 89 500 196  
    ```

另外，如果您愿意，也可以将边框的格式设置为: `<类别名称> <x1> <y1> <w> <h>`.

### 可选参数

可选参数:

| 参数 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 描述 | 样例 | 默认 |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	显示帮助信息 | `python pascalvoc.py -h` | |  
|  `-v`,<br>`--version` | 检查版本 | `python pascalvoc.py -v` | |  
| `-gt`,<br>`--gtfolder` | 包含真实边框文件的文件夹 | `python pascalvoc.py -gt /home/whatever/my_groundtruths/` | `/Object-Detection-Metrics/groundtruths`|  
| `-det`,<br>`--detfolder` | 包含检测边框文件的文件夹 | `python pascalvoc.py -det /home/whatever/my_detections/` | `/Object-Detection-Metrics/detections/`|  
| `-t`,<br>`--threshold` | IOU 阈值，判断检测结果是TP还是FP | `python pascalvoc.py -t 0.75` | `0.50` |  
| `-gtformat` | 真实边框的坐标值格式 [**\***](#asterisk) | `python pascalvoc.py -gtformat xyrb` | `xywh` |
| `-detformat` | 检测边框的坐标值格式 [**\***](#asterisk) | `python pascalvoc.py -detformat xyrb` | `xywh` | |  
| `-gtcoords` | 真实边框坐标值参考.<br>如果坐标值是相对于图片大小的相对值 (如YOLO中那样), 设置为 `rel`.<br>如果坐标是绝对值, 不依赖于图像大小, 设置为 `abs` |  `python pascalvoc.py -gtcoords rel` | `abs` |  
| `-detcoords` | 检测边框坐标值参考.<br>如果坐标值是相对于图片大小的相对值 (如YOLO中那样), 设置为 `rel`.<br>如果坐标是绝对值, 不依赖于图像大小, 设置为 `abs` | `python pascalvoc.py -detcoords rel` | `abs` |  
| `-imgsize ` | 图片尺寸的格式 `width,height` <int,int>.<br>如果 `-gtcoords` 或者 `-detcoords`需要， 设置为 `rel` | `python pascalvoc.py -imgsize 600,400` |  
| `-sp`,<br>`--savepath` | 保存结果的文件夹 | `python pascalvoc.py -sp /home/whatever/my_results/` | `Object-Detection-Metrics/results/` |  
| `-np`,<br>`--noplot` | 如果存在，则在执行过程中不显示任何绘图 | `python pascalvoc.py -np` | not presented.<br>Therefore, plots are shown |  

<a name="asterisk"> </a>
(**\***) 如果格式是 `<left> <top> <width> <height>`， 设置 `-gtformat xywh` 和 `-detformat xywh`. 
如果格式是 `<left> <top> <right> <bottom>`， 设置 `-gtformat xyrb` and/or `-detformat xyrb`.  

## 参考

* The Relationship Between Precision-Recall and ROC Curves (Jesse Davis and Mark Goadrich)
Department of Computer Sciences and Department of Biostatistics and Medical Informatics, University of
Wisconsin  
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

* The PASCAL Visual Object Classes (VOC) Challenge  
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf

* Evaluation of ranked retrieval results (Salton and Mcgill 1986)  
https://www.amazon.com/Introduction-Information-Retrieval-COMPUTER-SCIENCE/dp/0070544840  
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
