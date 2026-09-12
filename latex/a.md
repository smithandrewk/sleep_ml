# Introduction

Sleep is essential for optimal health [@cirelli_is_2008], yet more than
one third of the global population reports problems with sleep
[@jahrami_sleep_2021]. Preclinical sleep studies are imperative to
recapitulate phenotypes associated with poor sleep (hyperarousal,
cognitive impairment, slowed psychomotor vigilance, behavioral despair)
and uncover mechanisms contributing to psychopathology
[@graves_sleep_2003; @rentschler_reducing_2024; @siegel_sleep_2022; @wright_stress_2023].

Studies performed in rodents allow investigators to understand
underlying mechanisms that drive sleep homeostasis [@rayan_sleep_2024].
Importantly, vigilance states that compose sleep are evolutionarily
conserved amongst species, thereby rodent models can recapitulate the
unique polysomnographic characteristics of sleep states
[@rattenborg_evolution_2023]. The accurate determination of vigilance
stages into wake, rapid eye movement (REM) sleep, and non-REM (NREM) is
a critical step in analyzing acquired sleep-wake recordings. As REM and
NREM sleep serve functions uniquely different from a wake state,
correctly classifying vigilance states is imperative.

Currently, the gold standard for sleep staging (the determination of
sleep stages) in rodents is by human inspection of polysomnography
(PSG). PSG is a collection of signals obtained simultaneously during
sleep (for example, electroencephalography (EEG), electromyography
(EMG), electrooculography (EOG), etc.). Manual annotation of PSG by
human experts, however, is a time-consuming process that severely limits
efficiency. Further, human experts require extensive training that fails
to eliminate interscorer and intra-scorer variability
[@miladinovic_spindle_2019]. Therefore, developing a reliable,
reproducible, and accurate automated method for rodent sleep staging is
critical for more efficiently conducting preclinical sleep studies that
are critical to improving our understanding of sleep and health
outcomes.

Tethered rodents are combating confounds including limited mobility
within recording cages and potential impacts on natural sleep states
[@kramer_evaluation_2003; @papazoglou_non-restraining_2016; @rentschler_prenatal_2021].
For these reasons, it is advantageous to collect PSG via surgically
implanted telemetry transmitters. Due to the size and nature of
available telemetry transmitters for collecting PSG, the number of
signals acquired simultaneously during sleep is limited by number of
channels. Therefore, using a telemetry transmitter instead of tethering
rodents via wires decreases the number of obtained signals available for
sleep staging. Consistent with the principles of information theory,
data collected from fewer signals significantly increases the complexity
of downstream sleep staging [@shannon_mathematical_1948].

Several published algorithms only require a few signals that may be
collected via telemetry transmitters; however, they require a specific
combination of EEG, EMG, and EOG recordings
[@jha_slumbernet_2024; @vallat_open-source_2021]. Therefore, developing
an automated algorithm based on only a single-channel EEG signal is of
particular interest because of its versatility across hardware and
experimental settings. Further, automatic sleep staging based on a
single-channel EEG signal collected via a telemetry transmitter would
allow researchers the opportunity to investigate other signal types with
remaining available channels, such as electrocardiogram (EKG).

Deep learning has revolutionized various fields, including time series
analysis
[@smith_toward_2024; @chien_maeeg_2022; @vaswani_attention_2023; @he_deep_2015; @krizhevsky_imagenet_2012].
Advances in deep learning have garnered interest in developing automated
algorithms for sleep staging. Several such algorithms have been
published in recent years. Several recent advancements in automated
sleep staging are based on sleep studies in humans
[@supratak_deepsleepnet_2017; @vallat_open-source_2021; @perslev_u-time_2019; @Smith_Anand_Milosavljevic_Rentschler_Pocivavsek_Valafar_2022]
wherein sleep staging standards differ significantly from those in
rodent sleep studies [@rayan_sleep_2024]. Nearly all methods for
automatic sleep staging rely on multi-channel EEG recordings and derived
local field potential (LFP) signaling oscillations within the brain
[@miladinovic_spindle_2019; @gao_multiple_2016; @gross_open-source_2009; @stephenson_automated_2009; @louis_design_2004; @ellen_artificial_2021].
Of all the recently proposed deep learning algorithms for sleep staging,
we have only come across two that have attempted automatic sleep staging
in rodents via single-channel EEG
[@tezuka_real-time_2021; @liu_attention-based_2022].

In 2021, Tezuka et al. developed a CNN-LSTM neural network for sleep
staging in mice from only a single-channel EEG; however, the authors use
a private dataset, constrained to the light phase of the light-dark
cycle, and include Zeitgeber Time (ZT) and the Discrete Fourier
Transformation (DFT) as input to their model [@tezuka_real-time_2021].
The use of a private dataset limits researchers ability to confirm
results or compare novel methods. Constraining data to the light phase
of the light-dark cycle removes a dimension of variability that likely
reduces the complexity of sleep staging [@shannon_mathematical_1948].
Including ZT and DFT as input may introduce bias and limit
generalizability.

In 2022, Liu et al. developed a CNN in serial with an attention
mechanism for sleep staging in rodents from only a single-channel EEG
[@liu_attention-based_2022]. The authors use a public dataset from
previously published work [@miladinovic_spindle_2019] comprised of 22
24-hour EEG recordings totaling 528 and provide the results of 22-fold
leave-one-subject-out cross-validation [@liu_attention-based_2022]. Each
of the 22 folds is a 24-hour EEG recording and, therefore, consists of
both the light and the dark cycle. The authors, however, propose a
complex preprocessing to their single-channel EEG signal before being
input to their deep learning model including an array of hand-crafted
time-frequency transformations.

Many existing sleep staging algorithms face challenges such as being
proprietary, inadequately evaluated, dependent on complex data
preprocessing, reliant on multiple EEG channels, or requiring additional
signals such as EMG or EOG. Moreover, many are limited to human EEG data
or trained solely on publicly available datasets. In contrast, we
present an open-source, rigorously evaluated, end-to-end deep learning
model trained on a novel dataset for sleep staging in rodents using
single-channel EEG. Additionally, we contribute over 700 hours of
expert-scored sleep data, providing a valuable resource for future
research. We compare the performance of our model with closely related
methods, including those by Tezuka et al. (2021) and Liu et al. (2022).
Furthermore, using our newly introduced rodent dataset, termed
SleepyRats, we demonstrate that our model accurately predicts key sleep
architecture parameters, such as vigilance state durations, bout
numbers, and average bout durations.

# Methods

## Data Acquisition Procedure {#data-acquisition-procedure .unnumbered}

Adult male and female Wistar rats (n=16 for training; n=14 for
validation) were used in experiments in a facility fully accredited by
the Association for Assessment and Accreditation of Laboratory Animal
Care (AAALAC). Rats were kept on a 12 h/12 h light-dark cycle. All
protocols were approved by the Institutional Animal Care and Use
Committee (IACUC) at the University of South Carolina and were in
accordance with the National Institutes of Health Guide for the Care and
Use of Laboratory Animals.

Rats were implanted with EEG/EMG telemetry transmitters (PhysioTel
HD-S02, Data Sciences International (DSI), St. Paul, MN, USA). Briefly,
under isoflurane anesthesia, rats were placed in a stereotaxic frame
(Stoelting Co., Wood Dale, IL, USA). The telemetry transmitter was
implanted intraperitoneally through a dorsal incision of the abdominal
region. After an incision at the midline of the head was made, two EEG
leads were secured to two surgical stainless-steel screws (P1
Technologies, Roanoke, VA, USA) inserted into 0.5-mm burr holes at 2.0
mm anterior/+1.5 mm lateral and 7.0 mm posterior/-1.5 mm lateral
relative to bregma. The two EEG leads were secured using acrylic dental
cement (Stoelting Co., Wood Dale, IL, USA). Two EMG leads were inserted
into the dorsal cervical neck muscle about 1.0 mm apart and sutured into
place. The skin along the head was sutured, and animals were recovered
for a minimum of 7 days before sleep data acquisition.

Data used in training our model were obtained from rats that were
subject to 2 discrete sleep recording sessions each consisting of
exactly 24 hours [@milosavljevic_kynurenine_2023]. Sleep data were
acquired in a quiet, designated room where rats remained undisturbed for
the sleep recording duration on a PC running Ponemah 6.10 software
(DSI). EEG/EMG data were acquired with a sampling rate of 500 Hz.
Digitized signal data were exported for both recording sessions in
European Data Format (EDF), a standard file type for raw PSG. An expert
in sleep staging manually annotated each recording with sleep stages in
10-second epochs offline using NeuroScore 3.0 software (DSI).

For each of the 2 recording sessions, 16 discrete 24-hour EEG/EMG
signals resulted from data collection (one for each rat). We group the
pairs of 24-hour PSG recordings for each rat into a 48-hour PSG
recording and ensure that data is processed in a temporally consistent
manner. It is however critical that we maintain rat identity throughout
the entire investigation for proper training and evaluation of our
algorithm. Therefore, we refer to the shape of the dataset as 16 rats x
48-hour PSG recording, as shown in
Fig. [\[dataset\]](#dataset){reference-type="ref" reference="dataset"}.

## Validation {#validation .unnumbered}

The validation paradigm is critical for proper model evaluation and
establishing robust results [@Goodfellow-et-al-2016]. The goal of
defining a validation paradigm is to simulate the use of a model
deployed in the real world. Most machine learning applications aim to
develop a system capable of generalization as opposed to memorization.
Generalization refers to a model's ability to perform well when provided
new data that it was not explicitly trained on, whereas memorization
refers to a model's tendency to excessively rely on specific details and
examples from the training data, failing to capture underlying patterns
or relationships. Because of a deep learning model's propensity to
overfit to training data, leakage of information between training and
testing sets may result in overoptimistic results. Subject-based
datasets may be prone to leakage of information due to systematic
variation between subjects (exposure of a network to any given subject
in both training and testing). To remove the possibility of information
leakage, we use a cross-validation scheme (a particular form of
validation in general) where a given rat's PSG data is contained within
only one-fold. This is referred to as leave-one-out cross-validation, a
special case of cross-validation, where the number of folds is equal to
the number of groups. Since there are 16 rats in our dataset, we define
each rat (consisting of a 48-hour PSG recording) as a group. Therefore,
for each fold, we train a model on the 15 others and evaluate the last
rat. This approach typically provides the most reliable validation of
generalization under the condition of data scarcity
[@Hastie_Tibshirani_Friedman_2009].

## Model {#model .unnumbered}

We designed our model to independently encode EEG epochs into
information-rich low-dimensional vectors then incorporate sequential
information from neighboring epochs to make a single sleep stage
prediction. To accomodate this design, we combine a convolutional neural
network [@krizhevsky_imagenet_2012] with a Long-Short Term Memory neural
network [@hochreiter_long_1997]. We designed our model to take a
sequence of 9 10-second epochs as input and precit the sleep stage of
the center (5th) epoch based on the common practice of our expert
labelers; however, one could easily adapt the architecture to take any
sequence length. Our model architecture is shown in
Fig. [1](#model){reference-type="ref" reference="model"}.

![ **Model diagram of our algorithm.** The model accepts a sequence of 9
10-second epochs to predict the class of the center 10-second epoch
(epoch i, as denoted in the figure) Each 10-second epoch in the sequence
is independently encoded and then forwarded to a Long Short-Term Memory
(LSTM) neural network to share information between neighboring 10-second
epochs. The 64-dimensional LSTM output vector is linearly projected down
to three dimensions and mapped to a probability distribution over 3
values (corresponding to 3 vigilant states) by the Softmax function.
](fig1.pdf){#model width="\\textwidth"}

**Epoch Encoder:**During the first stage of the network, each 10-second
epoch in the input sequence is independently encoded by the epoch
encoder, shown in Fig. [1](#model){reference-type="ref"
reference="model"}. The epoch encoder is a RegNet
[@radosavovic_designing_2020] modern variant of the convolutional neural
network [@krizhevsky_imagenet_2012] that investigated optimal design of
Residual Neural Networks [@he_deep_2015]. The epoch encoder consists of
a stack of 3 identical residual blocks [@he_deep_2015] and a final
global average pooling [@lin_network_2014], except that the number of
convolution channels vary. The number of convolution channels is shown
within each residual block in
Fig. [\[model_components\]](#model_components){reference-type="ref"
reference="model_components"}. Each residual block consists of 3
sequences of convolution, layer normalization [@ba_layer_2016], and ReLU
nonlinearity [@fukushima_visual_1969], as well as an elementwise
residual connection before the final ReLU. The convolution kernels for
each residual block are size 8, 5, and 3, in that order.

**LSTM and Classifier Head:**The resultant sequence of spatially encoded
feature maps is forwarded to a temporally aware neural network,
resulting in a single output feature map that integrates spatial and
temporal information. Lastly, this spatio-temporally-encoded feature map
is presented to a final classifier that maps 3 output states onto a
probability distribution, each of which corresponds to one of 3 vigilant
states. Additional details regarding the network architecture can be
found in the supplementary material
Fig. [\[model_components\]](#model_components){reference-type="ref"
reference="model_components"}.

## Preparing Dataset For Training {#preparing-dataset-for-training .unnumbered}

We train the model to accept an input sequence of 9 10-second epochs and
output a probability distribution over 3 classes corresponding to 3
stages (paradoxical sleep, slow-wave sleep, and wakefulness) for the
10-second epoch in the center of the sequence. To allow the evaluation
of the first 4 10-second epochs, we pad each 1-dimensional EEG signal
(one 24-hour PSG recording) with 4 10-second epochs with zero value on
each end. Each zero-padded PSG signal is partitioned into 10-second
epochs and transformed using a simple moving window of 9 10-second
epochs with a stride of 1. Initial expert sleep staging remains
unchanged; therefore, each window of 9 sequential 10-second epochs is
annotated by the sleep stage of the epoch in the center of the window as
desired. We take a 5% of the training set on each fold for validation.

## Optimization {#optimization .unnumbered}

We trained our model using mini batches of size 32 epochs (input
sequences of 9 10-second epochs and the corresponding sleep staging)
until validation loss did not improve for 30 epochs. We perform
hyperparameter optimization using the validation loss. During training,
we used cross-entropy [@kullback_information_1951] to compare model
predictions to reference sleep staging. For model weight optimization,
we used Adam [@kingma_adam_2017]. All the pre and post data processing
steps were completed in Python using PyTorch as the sole machine
learning platform.

## Evaluation Metrics {#evaluation-metrics .unnumbered}

Defining good evaluation metrics is crticial for establishing proper
results in deep learning [@Goodfellow-et-al-2016]. Sleep staging is
inherently an imbalanced classification problem. Using accuracy as a
performance metric in an imbalanced classification problem provides
overoptimistic results. A good performance metric for evaluating an
imbalanced classification is F1-score, the harmonic mean between
precision and recall. In a classification setting with multiple classes,
we define the F1 Score for a given class $C$ as $$\begin{gathered}
\text{F1 Score}_C = \frac{2 (\text{precision}_C)(\text{recall}_C)}{\text{precision}_C + \text{recall}_C},
\text{precision} = \frac{\text{TP}_C}{\text{TP}_C+\text{FP}_C},\text{recall} = \frac{\text{TP}_C}{\text{TP}_C+\text{FN}_C}
\end{gathered}$$ where TP is the number of true positives, FP the number
of false positives, and FN the number of false negatives. We define the
macro F1 Score as the average F1 Score over each class
$$\frac{\Sigma_C \text{F1 Score}_C}{N}$$ where there are $N$ classes.

## Comparison Against Previous Work {#comparison-against-previous-work .unnumbered}

In addition to testing the developed system using the procedure stated
in section 3.2, we further tested the performance of our neural network
against a recently developed sleep-staging algorithm published by Liu et
al.[@liu_attention-based_2022]. Their algorithm is based on a deep
learning approach that was developed and evaluated using a publicly
available dataset [@miladinovic_spindle_2019]. We evaluated our
algorithm using the same dataset to provide a direct comparison of
performance and to establish the generalization capabilities of our
trained network.

The validation scheme for our dataset is comparable to the validation
approach in Liu et al.[@liu_attention-based_2022], wherein a
leave-one-out cross-validation over 22 folds, each corresponding to a
24-hour sleep recording from a unique rodent, was utilized. The SPINDLE
dataset is composed of multiple cohorts of animals sampled at varying
sampling rates (128 Hz, 200 Hz, and 512 Hz) [@miladinovic_spindle_2019].
To adapt the SPINDLE dataset to our model, we resampled each EEG
recording to 500 Hz using the Fourier Transform method and performed the
same zero-padding technique described above (see 3.4 Data Preparation).
Additionally, the SPINDLE dataset defines the length of an epoch as 4
seconds and performs the classification at that resolution. Our model
outputs sleep staging at the 10-second resolution; therefore, our
predictions are not directly comparable to the SPINDLE reference. To
alleviate this problem, we resampled our prediction signal from the
10-second resolution to the 4-second resolution (i.e. up sampling). By
up sampling our prediction signal (as opposed to down sampling the
SPINDLE reference signal), no information is lost from the SPINDLE
reference signal and no information is gained in the prediction signal;
therefore, the problem's complexity remains unchanged. Up sampling a
signal from 0.1 Hz to 0.25 Hz is non-trivial since 0.1 does not divide
0.25, resulting in 1 unaligned reference 4-second epoch for every 2
predictive 10-second epochs as shown in Fig. 4. For every 2 consecutive
10-second epochs in our prediction signal, there are 5 consecutive
4-second epochs in the SPINDLE reference signal. The first two and last
two reference 4-second epochs align within 10-second epochs in our
prediction signal and are directly comparable. However, every third
4-second epoch in the SPINDLE reference signal spans 2 10-second epochs
in our prediction signal. To account for this, we compare this SPINDLE
reference 4-second epoch to both predictive 10-second epochs which it
spans and take the mean of the metric being computed. For example, the
third 4-second epoch in the SPINDLE reference signal (P) spans 2
10-second epochs in our prediction signal (P and S, respectively);
therefore, this epoch contributes 5 to accuracy (Fig.
[\[spindle\]](#spindle){reference-type="ref" reference="spindle"}A).

## Validation of conclusions on vigilance state duration and architecture {#validation-of-conclusions-on-vigilance-state-duration-and-architecture .unnumbered}

Data used in validating our model for conclusions on vigilance state
duration and architecture were obtained from rats that were subject to
discrete sleep recording sessions, as described above. An expert in
sleep staging manually annotated each recording with sleep stages in
10-second epochs offline using NeuroScore 3.0 software (DSI) for the
first 4-hour PSG recording. Comparisons were made between LSTM scoring
the human scoring in 1-hour bins from ZT 0 to 4 evaluating vigilance
state duration, bout number, and average duration of each bout.
Comparisons were made between LSTM scoring and human scoring using a
two-way repeated measures (RM) analysis of variance (ANOVA) with ZT and
scoring as within-subject factors. Post hoc analysis was conducted to
evaluate changes across time with Dunnett test compared to ZT 0. All
statistical analyses were performed using Prism 9.0 (GraphPad Software,
La Jolla, CA, USA) and significance was defined as P \< 0.05.

## Data and Code Availability {#data-and-code-availability .unnumbered}

All data can be requested from our website
(<https://ifestos.cse.sc.edu/datasets/ekyn.tar.gz>). The SPINDLE dataset
can be found at <https://sleeplearning.ethz.ch/paper/>. The source code
and documentation of the algorithm are available at
<https://github.com/smithandrewk/ml-sleep>. The Python code to reproduce
all the results and figures of this paper can be found at
<https://github.com/smithandrewk/ml-sleep>. All analyses were conducted
in Python 3.11 using PyTorch 1.13.1.

# Results

## KYNA Dataset {#kyna-dataset .unnumbered}

The dataset consisted of 32 unique 24-hour PSG recordings obtained from
a cohort of 16 rats subjected to 2 discrete sleep recording sessions. We
trained our model with fixed architecture and hyperparameters over all
16 folds (where each fold corresponds to an individual rat). The
training set, for each of the 16 folds, consisted of 30 24-hour EEG
signals (259200 10-second epochs). A small portion of the training set
(5%) was taken to implement early stopping for model training . Early
stopping is a regularization technique that monitors the model's
performance on a validation set and halts training once performance on
the validation set starts to degrade, thereby preventing overfitting and
improving generalization to unseen data. The testing set, for each of
the 16 folds, consisted of 2 24-hour EEG signals (17280 10-second
epochs). The overall performance of the algorithm is described in
Fig. [2](#ekyn_grid){reference-type="ref" reference="ekyn_grid"}A. The
box plot shows the distribution of performance over 16 folds of
cross-validation. The mean f1-score, calculated across all 16 folds from
leave-one-out cross-validation, was 87.6%. One outlier incurred several
misclassifications, resulting in a recall of approximately 78% and an
f1-score of approximately 83% (Fig. [2](#ekyn_grid){reference-type="ref"
reference="ekyn_grid"}A).

![ **Performance of the algorithm on our dataset created for the model
training over 16 folds of cross-validation.**(A) Precision, recall, and
f1-score over 16 folds of cross-validation are shown by a boxplot. (B)
Dimensionality reduction of EEG encodings colored by sleep stage. Our
network encodes EEG signals to be well-separated and therefore serves as
the basis for accurate sleep staging. (C) Confusion matrix. The diagonal
elements represent the percentage of 10-second epochs that were
correctly classified by the algorithm (recall), whereas the off-diagonal
elements show the percentage of 10-second epochs mislabeled by the
algorithm. (D) Duration of each sleep stage for predicted signal and
reference signal over 16 folds of cross-validation expressed as a
proportion of the total duration of the given EEG recording. Pairs of
boxplots are shown for each stage where the left box depicts the
predicted distribution, and the right box depicts the reference
distribution. P - paradoxical sleep, S - slow-wave sleep, W -
wakefulness. ](fig2.pdf){#ekyn_grid width="\\textwidth"}

:::::: centering
::::: small
:::: sc
::: {#kyna_table}
  Class          F1             recall         precision     
  ------- ---------------- ---------------- ---------------- --
  P        78.2$\pm$ 14.0   81.1$\pm$ 16.6   78.5$\pm$ 12.5  
  S        92.3$\pm$ 5.0    91.2$\pm$ 9.4    94.1$\pm$ 3.8   
  W        92.2$\pm$ 2.9    93.1$\pm$ 3.8    91.8$\pm$ 6.4   

  : Performance of our model on our dataset broken down by class.
  Metrics are the mean and standard deviation over 16 folds of
  cross-validation for f1-score, recall, and precision.
:::
::::
:::::
::::::

The model compresses 45,000-dimensional (500 Hz \* 10 Seconds \* 9
epochs) raw EEG inputs into a 64-dimensional latent spatiotemporal
representation before being linearly projected into three dimensions for
sleep staging. To visualize similarities between these 64-dimensional
feature maps, we show the results of tSNE [@maaten_visualizing_2008], a
dimension reduction algorithm designed to preserve high-dimensional
neighborhoods in Fig. [2](#ekyn_grid){reference-type="ref"
reference="ekyn_grid"}B. As expected, slow-wave sleep and wakefulness
stages are well-clustered whereas many wake and paradoxical sleep
stages' embeddings escape into the other's neighborhood. The similarity
between EEG waveforms during wakefulness and paradoxical sleep almost
certainly contributes to these mixtures.

We tested the sleep staging performance of individual vigilant states
(paradoxical sleep, slow-wave sleep, and wakefulness) by computing a
recall confusion matrix (Fig. [2](#ekyn_grid){reference-type="ref"
reference="ekyn_grid"}C). The average recall for paradoxical sleep is
81.1%, and most misclassifications were classified as wakefulness. Both
slow-wave sleep and wakefulness have a recall over 91.2%. Class specific
recall, precision, and f1 score can be found in
Table [1](#kyna_table){reference-type="ref" reference="kyna_table"}.

The dynamics of the composition of sleep, in terms of sleep stages, are
complex but important for interpretations of sleep quality in
preclinical studies. A few important parameters of sleep composition,
beyond merely total duration, can thus serve as a unique characteristic
for evaluation of model performance and include total bout duration ,
average bout duration, and number of bouts (all distributed by vigilant
state). We investigated these parameters
(Fig. [\[proxy_grid\]](#proxy_grid){reference-type="ref"
reference="proxy_grid"}) and show the total duration of each stage as a
proportion of EEG recording time in
Fig. [2](#ekyn_grid){reference-type="ref" reference="ekyn_grid"}D.

Tezuka et. al [@tezuka_real-time_2021] report a mean f1-score of 80.8
across 10 folds of cross-validation, which is notably lower than the
F1-score achieved by our model. Consequently, we focus our comparison on
SPINDLE, where the performance gap is more meaningful for further
evaluation.

## SPINDLE Dataset {#spindle-dataset .unnumbered}

It is critical to emphasize that our model was optimized for sleep
staging with 10-second epochs and that stages in the SPINDLE dataset
were scored with 4-second epochs. In evaluating our model on the SPINDLE
dataset, we find that, even by unique characteristic of 10-second
epochs, we outperform the algorithm proposed by Liu et al.
[@liu_attention-based_2022] despite theirs being optimized specifically
for the SPINDLE dataset . The results of 22 folds leave-one-out
cross-validation on the SPINDLE dataset are reported in
Table [\[metrics_spindle\]](#metrics_spindle){reference-type="ref"
reference="metrics_spindle"}. Notably, using the same dataset and
validation scheme, our model achieves a mean f1-score of 89.6% compared
to 88.1% in Liu et al. [@liu_attention-based_2022]. Though we achieve
higher performance in all three classes, most of the performance
increase comes from the paradoxical sleep class where our model achieves
f1-score of 83.6% as opposed to 79.4% in Liu et al.
[@liu_attention-based_2022]. It is critical to emphasize that our entry
in Table [\[metrics_spindle\]](#metrics_spindle){reference-type="ref"
reference="metrics_spindle"} reflects our performance after up sampling
our model's 10-second prediction signal to be a 4-second prediction
signal. We performed further analysis on the distribution of performance
across the 22 folds of cross-validation, the results of which are shown
in Figure [\[spindle_grid\]](#spindle_grid){reference-type="ref"
reference="spindle_grid"}.

Notably, one outlier achieves an f1-score of approximately 83% . To
further understand model performance broken down by class, we present
the mean recall confusion matrix over the 22 folds of cross-validation
(Fig. [\[spindle_grid\]](#spindle_grid){reference-type="ref"
reference="spindle_grid"}C). Most misclassifications oc slackcur between
paradoxical sleep and wakefulness, as expected due to the similarity
between their EEG waveforms. For each input sample, our model outputs a
probability distribution over all possible stages; therefore, we can
quantify the model's confidence in predicting a certain class. For each
fold, we take the average confidence of the model and compare this to
the macro f1-score for that recording
(Fig. [\[spindle_grid\]](#spindle_grid){reference-type="ref"
reference="spindle_grid"}B). Linear regression between these two
variables suggests a positive relationship between a model's average
confidence and the model's performance. As presented for the dataset
created for the model training, we investigated the proportion of
recording time that each stage constituted as well as the proportion of
total EEG recording time predicted by the model
(Fig. [\[spindle_grid\]](#spindle_grid){reference-type="ref"
reference="spindle_grid"}D). Though for each class and each parameter
the distribution of the SPINDLE reference values and predicted values is
comparable, it is still unclear how well the model performs for each EEG
recording. Thus, we perform linear regression between the SPINDLE
reference values and predicted values for all combinations of classes
and all three parameters
(Fig. [\[proxy_grid\]](#proxy_grid){reference-type="ref"
reference="proxy_grid"}). Strong positive linear correlation was
confirmed for all combinations of classes and parameters. If the model
was a perfect predictor, all individual data points would fall on the x
= y line. A common method of visualizing sleep stages for a given EEG
recording is a hypnogram. The reference hypnogram is shown in
Fig. [\[hypnogram\]](#hypnogram){reference-type="ref"
reference="hypnogram"}A and our model's predicted hypnogram in shown in
Fig. [\[hypnogram\]](#hypnogram){reference-type="ref"
reference="hypnogram"}B. Further, the EEG signal that constitutes the
only input to the model, as it is an end-to-end deep learning model, is
shown in Fig. [\[hypnogram\]](#hypnogram){reference-type="ref"
reference="hypnogram"}C. Lastly, the predicted probability distribution
over all possible stages by the model for the EEG signal is shown in
Fig. [\[hypnogram\]](#hypnogram){reference-type="ref"
reference="hypnogram"}D. We take the highest probability as the
predicted class, and the minimum confidence with which our model could
predict a certain class is slightly above 0.33. Hypothetically, higher
average confidence values result in higher network performance. We
notice that the confidence level of the network fluctuates throughout
the recording; in longer bouts, confidence seems to increase, and in
shorter bouts, confidence seems to decrease. Region from epoch 150 to
epoch 250 on Fig. [\[hypnogram\]](#hypnogram){reference-type="ref"
reference="hypnogram"}D shows where the network is almost 100%
confident.

## SleepyRat Dataset {#sleepyrat-dataset .unnumbered}

SleepyRat study was conducted in a cohort of rats to evaluate our novel
sleep scoring network. We evaluated vigilance state durations and
architecture comparing the findings of LSTM scored date to an expert
human. Taken together, we determined no significant differences in the
data due to scoring (Table [2](#architecture_table){reference-type="ref"
reference="architecture_table"}), but found significant impact of time
of day, ZT, on vigilance state duration and architecture (bout number
and average duration of vigilance bouts) parameters. In the parameters
that a main effect of ZT was determined, post hoc Dunnett test analysis
revealed significant differences between ZT 0-1 and subsequent time
bins. Of note, findings between LTSM scoring and human scoring on the
impact of time on vigilance state duration and architecture parameters
aligned (Figure [3](#architecture){reference-type="ref"
reference="architecture"}).

[]{#architecture_table label="architecture_table"}

::: {#architecture_table}
  **Vigilance State**          **Scoring**              **ZT**              **Interaction**
  --------------------------- ------------- ------------------------------ -----------------
  **Duration**                                                             
  S, NREM                        0.4852                0.0423\*                 0.3153
  P, REM                         0.1281      $<$`<!-- -->`{=html}0.0001\*       0.2553
  W, Wake                        0.3713                0.0008\*                 0.2074
  **Bout Number**                                                          
  S, NREM                        0.9327                0.0341\*                 0.0516
  P, REM                         0.1857      $<$`<!-- -->`{=html}0.0001\*       0.4208
  W, Wake                        0.8179                 0.0535                  0.0876
  **Average Bout Duration**                                                
  S, NREM                        0.4614                 0.1865                  0.2871
  P, REM                         0.3060                 0.4495                  0.3416
  W, Wake                        0.4412                0.0023\*                 0.0624

  : **Results of SleepyRat dataset statistical analysis comparing LSTM
  scoring and human expert scoring.**P values are indicated for main
  effects of scoring, ZT, and interaction between scoring and ZT. \*
  indicates significant effect wherein P $<$ 0.05. P values of two-way
  RM ANOVA.
:::

![ **Sleep-wake duration and architecture data do not significantly
differ between LSTM and human expert scoring of SleepyRat dataset.** (A)
NREM duration, (B) REM duration, (C) Wake duration, (D) NREM bout
number, (E) REM bout number, (F) Wake bout number, (G) average NREM bout
duration, (H) average REM bout duration, (I) average Wake bout duration.
Two-way RM ANOVA with post-hoc Dunnett test. \* P \< 0.05, \*\* P \<
0.01, \*\*\* P \< 0.001, \*\*\*\* P \< 0.0001. Data are mean $\pm$ SEM.
N =14 per group. ](fig3.pdf){#architecture width="\\textwidth"}

# Discussion

The current gold standard for sleep staging involves a trained human
expert manually annotating polysomnography (PSG), typically using EEG
and EMG signals, to classify sleep stages. Developing a reliable,
automated algorithm for sleep staging is critical for improving the
efficiency of EEG signal processing. An automated method that uses only
single-channel EEG recordings would provide a significant advancement by
reducing the need for multi-channel recordings and manual annotations.

Human experts, after extensive training, rely on visual inspection of
signals (such as EEG and EMG) and feature-based tools (like
power-spectral density) to classify sleep epochs into vigilant stages.
The patterns within these signals (e.g., amplitude and frequency)
ultimately guide expert classifications. To replicate this process, we
implemented a convolutional neural network to encode visual patterns
contained in raw EEG signals down to a representation more amenable to
classification. Each 10-second epoch is processed independently,
resulting in a set of feature maps.

To capture temporal relationships between EEG patterns, we introduced a
\"temporal encoder\" following the spatial encoder. This temporal
encoder is designed to share information across time by considering 4
preceding and 4 succeeding epochs (for a total window of 9 epochs, or 90
seconds) centered around the target epoch to be classified. This mirrors
the 90-second window used by human scorers when analyzing PSG. The
temporal encoder is based on a Long Short-Term Memory (LSTM) network,
which is capable of learning and integrating temporal dependencies from
the data. The combination of spatial and temporal encoders allows our
model to produce an intermediate representation rich in spatiotemporal
information.

Not only does our model effectively capture spatiotemporal information,
but it also ensures that the representations of different sleep stages
are well-separated in the latent space, which is key for accurate
classification. This capability of deep learning---to automatically
transform raw data into meaningful representations for downstream
tasks---highlights the strength of our approach. Our end-to-end model
takes raw single-channel EEG signals as input and outputs sleep stage
classifications, avoiding the need for feature extraction steps that
could introduce bias or redundancy.

To further validate the usability of our model, we conducted an
extensive evaluation using the SleepyRats dataset. This dataset consists
of over 700 hours of single-channel EEG recordings from a cohort of 16
rats recorded across two 24-hour sleep sessions. The dataset provides a
robust foundation for evaluating the model's performance in terms of
vigilance state durations, bout numbers, and the average duration of
bouts. A two-way repeated measures ANOVA was performed to compare the
LSTM-based sleep staging results with human expert scoring. No
significant differences were found between the LSTM predictions and
human annotations for sleep stage durations and architecture parameters,
confirming the model's reliability for preclinical sleep analysis. As
shown in Figure [3](#architecture){reference-type="ref"
reference="architecture"}, the consistency in the number of bouts and
the average bout durations across different sleep stages between LSTM
scoring and human scoring further validates the model's effectiveness.

Additionally, we provide comparisons to existing datasets like SPINDLE,
which consisted of 22 folds and under 500 hours of EEG recordings.
Proper evaluation of a deep learning model becomes the most critical
step in the development of machine learning systems. Our model was
rigorously evaluated using leave-one-out cross-validation on both the
newly introduced SleepyRats dataset and the SPINDLE dataset. In total,
38 deep learning models were developed during the validation phase, and
performance was measured using F1 score, precision, and recall.

Through the introduction of the SleepyRats dataset and the validation
performed using ANOVA, we have demonstrated that our deep learning model
not only automates sleep staging effectively but also provides reliable
estimates of sleep architecture parameters. This validation, combined
with the model's performance on cross-validation, ensures its usability
for large-scale preclinical sleep analysis, offering a significant
reduction in the time and resources required for manual sleep staging.

# Acknowledgements {#acknowledgements .unnumbered}

Thanks to reviewers who gave useful comments, to colleagues who
contributed to the ideas, and to funding agencies that provided
financial support.

# Author Contributions {#author-contributions .unnumbered}

Andrew Smith contributed to the conception and design of the study, data
analysis and interpretation, and drafting the manuscript. Courtney
Wright contributed to data acquisition and provided revisions. Charlie
Grant contributed to data analysis and revisions. Snezana Milosavljevic
contributed to data acquisition and revisions. Ana Pocivavsek
contributed to the conception and design of the study, provided
revisions, and served as an advisor. Homayoun Valafar contributed to the
conception and design of the study, provided revisions, and served as an
advisor. All authors reviewed and approved the final manuscript, and
agree to be accountable for all aspects of the work.

# Funding {#funding .unnumbered}

Research reported in this publication was supported by funds awarded to
Dr. Valafar from the National Institute of General Medical Sciences of
the National Institutes of Health under Award Number P20GM103499.
Research reported in this public was also supported by R01 NS 102209 and
R21 AG080335 awarded to Dr. Ana Pocivavsek.

# Competing Interests {#competing-interests .unnumbered}

The authors have nothing to disclose.

# Figure Legends {#figure-legends .unnumbered}

**Figure 1. Model diagram of our algorithm.**The model accepts a
sequence of 9 10-second epochs to predict the class of the center
10-second epoch (epoch i, as denoted in the figure) Each 10-second epoch
in the sequence is independently encoded and then forwarded to a Long
Short-Term Memory (LSTM) neural network to share information between
neighboring 10-second epochs. The 64-dimensional LSTM output vector is
linearly projected down to three dimensions and mapped to a probability
distribution over 3 values (corresponding to 3 vigilant states) by the
Softmax function.

**Figure 2. Performance of the algorithm on our dataset created for the
model training over 16 folds of cross-validation.**(A) Precision,
recall, and f1-score over 16 folds of cross-validation are shown by a
boxplot. (B) Dimensionality reduction of EEG encodings colored by sleep
stage. Our network encodes EEG signals to be well-separated and
therefore serves as the basis for accurate sleep staging. (C) Confusion
matrix. The diagonal elements represent the percentage of 10-second
epochs that were correctly classified by the algorithm (recall), whereas
the off-diagonal elements show the percentage of 10-second epochs
mislabeled by the algorithm. (D) Duration of each sleep stage for
predicted signal and reference signal over 16 folds of cross-validation
expressed as a proportion of the total duration of the given EEG
recording. Pairs of boxplots are shown for each stage where the left box
depicts the predicted distribution, and the right box depicts the
reference distribution. P - paradoxical sleep, S - slow-wave sleep, W -
wakefulness.

**Figure 3. Sleep-wake duration and architecture data do not
significantly differ between LSTM and human expert scoring of SleepyRat
dataset.**(A) NREM duration, (B) REM duration, (C) Wake duration, (D)
NREM bout number, (E) REM bout number, (F) Wake bout number, (G) average
NREM bout duration, (H) average REM bout duration, (I) average Wake bout
duration. Two-way RM ANOVA with post-hoc Dunnett test. \* P \< 0.05,
\*\* P \< 0.01, \*\*\* P \< 0.001, \*\*\*\* P \< 0.0001. Data are mean
$\pm$ SEM. N =14 per group.
