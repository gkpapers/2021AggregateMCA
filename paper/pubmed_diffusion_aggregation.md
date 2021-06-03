((diffusion MRI) OR (diffusion weighted imaging) OR (dwi) OR (dMRI) OR (d-MRI) OR (diffusion Imaging)) AND (brain) AND ((network) OR (connectome) OR (structural connectome)) AND ((data augmentation) OR (dataset augmentation))

| Paper Number | Yr   | Proposes Augmentation | Uses Existing Augmentation | DL   | N  Subjs | Relevant                                                    |
| ------------ | ---- | --------------------- | -------------------------- | ---- | -------- | ----------------------------------------------------------- |
| 1            | 2020 | No                    | Yes                        | Yes  | 176      | No: avoiding diffusion, T1-> age prediction                 |
| 2            | 2012 | No                    | No                         | No   | N/A      | No: review for fibromialgia                                 |
| 3            | 2017 | Yes                   | No                         | Yes  | N/A      | No: transfer learning from HCP to worse data                |
| 4            | 2017 | No                    | No                         | No   | 15       | No: surgical assist with structural networks                |
| 5            | 2018 | Yes                   | No                         | Yes  | 106      | No: Lung disease from HRCT images                           |
| 6            | 2020 | No                    | No                         | No   | N/A      | No: studying tract profiles in HCP                          |
| 7            | 2016 | No                    | No                         | No   | N/A      | No: pig brain atlas                                         |
| 8            | 2016 | No                    | No                         | No   | 21       | No: temporal lobe epilepsy structure-function relationships |
| 9            | 2021 | Yes                   | No                         | Yes  | 48       | Yes: classification in MS, F1 from 66 -> 81                 |
| 10           | 2019 | Yes                   | No                         | Yes  | 43       | No: segmetation of MR perfusion in stroke                   |
| 11           | 2021 | No                    | No                         | Yes  | N/A      | No: segmentation of DWI data from HCP                       |
| 12           | 2013 | No                    | No                         | No   | N/A      | No: TBI and functional connectivity                         |
| 13           | 2016 | No                    | No                         | No   | N/A      | No: structure-function relationships                        |



1. Front Neurol. 2020 Oct 19;11:584682. doi: 10.3389/fneur.2020.584682. eCollection 
  2020.

Brain Age Prediction of Children Using Routine Brain MR Images via Deep 
Learning.

Predicting brain age of children accurately and quantitatively can give help in 
brain development analysis and brain disease diagnosis. Traditional methods to 
estimate brain age based on 3D magnetic resonance (MR), T1 weighted imaging 
(T1WI), and diffusion tensor imaging (DTI) need complex preprocessing and extra 
scanning time, decreasing clinical practice, especially in children. This 
research aims at proposing an end-to-end AI system based on deep learning to 
predict the brain age based on routine brain MR imaging. We spent over 5 years 
enrolling 220 stacked 2D routine clinical brain MR T1-weighted images of healthy 
children aged 0 to 5 years old and randomly divided those images into training 
data including 176 subjects and test data including 44 subjects. Data 
augmentation technology, which includes scaling, image rotation, translation, 
and gamma correction, was employed to extend the training data. A 10-layer 3D 
convolutional neural network (CNN) was designed for predicting the brain age of 
children and it achieved reliable and accurate results on test data with a mean 
absolute deviation (MAE) of 67.6 days, a root mean squared error (RMSE) of 96.1 
days, a mean relative error (MRE) of 8.2%, a correlation coefficient (R) of 
0.985, and a coefficient of determination (R 2) of 0.971. Specially, the 
performance on predicting the age of children under 2 years old with a MAE of 
28.9 days, a RMSE of 37.0 days, a MRE of 7.8%, a R of 0.983, and a R 2 of 0.967 
is much better than that over 2 with a MAE of 110.0 days, a RMSE of 133.5 days, 
a MRE of 8.2%, a R of 0.883, and a R 2 of 0.780.




2. Curr Pain Headache Rep. 2012 Oct;16(5):388-98. doi: 10.1007/s11916-012-0284-9.

Brain imaging in fibromyalgia.

Fibromyalgia is a primary brain disorder or a result of peripheral dysfunctions 
inducing brain alterations, with underlying mechanisms that partially overlap 
with other painful conditions. Although there are methodologic variations, 
neuroimaging studies propose neural correlations to clinical findings of 
abnormal pain modulation in fibromyalgia. Growing evidences of specific 
differences of brain activations in resting states and pain-evoked conditions 
confirm clinical hyperalgesia and impaired inhibitory descending systems, and 
also demonstrate cognitive-affective influences on painful experiences, leading 
to augmented pain-processing. Functional data of neural activation abnormalities 
parallel structural findings of gray matter atrophy, alterations of intrinsic 
connectivity networks, and variations in metabolites levels along multiple 
pathways. Data from positron-emission tomography, 
single-photon-emission-computed tomography, blood-oxygen-level-dependent, 
voxel-based morphometry, diffusion tensor imaging, default mode network 
analysis, and spectroscopy enable the understanding of fibromyalgia 
pathophysiology, and favor the future establishment of more tailored treatments.




3. Neuroimage. 2017 May 15;152:283-298. doi: 10.1016/j.neuroimage.2017.02.089. Epub 
2017 Mar 3.

Image quality transfer and applications in diffusion MRI.

This paper introduces a new computational imaging technique called image quality 
transfer (IQT). IQT uses machine learning to transfer the rich information 
available from one-off experimental medical imaging devices to the abundant but 
lower-quality data from routine acquisitions. The procedure uses matched pairs 
to learn mappings from low-quality to corresponding high-quality images. Once 
learned, these mappings then augment unseen low quality images, for example by 
enhancing image resolution or information content. Here, we demonstrate IQT 
using a simple patch-regression implementation and the uniquely rich diffusion 
MRI data set from the human connectome project (HCP). Results highlight 
potential benefits of IQT in both brain connectivity mapping and microstructure 
imaging. In brain connectivity mapping, IQT reveals, from standard data sets, 
thin connection pathways that tractography normally requires specialised data to 
reconstruct. In microstructure imaging, IQT shows potential in estimating, from 
standard "single-shell" data (one non-zero b-value), maps of microstructural 
parameters that normally require specialised multi-shell data. Further 
experiments show strong generalisability, highlighting IQT's benefits even when 
the training set does not directly represent the application domain. The concept 
extends naturally to many other imaging modalities and reconstruction problems.




4. Brain. 2017 Mar 1;140(3):641-654. doi: 10.1093/brain/awx004.

Individual brain structure and modelling predict seizure propagation.

See Lytton (doi:10.1093/awx018) for a scientific commentary on this 
article.Neural network oscillations are a fundamental mechanism for cognition, 
perception and consciousness. Consequently, perturbations of network activity 
play an important role in the pathophysiology of brain disorders. When 
structural information from non-invasive brain imaging is merged with 
mathematical modelling, then generative brain network models constitute 
personalized in silico platforms for the exploration of causal mechanisms of 
brain function and clinical hypothesis testing. We here demonstrate with the 
example of drug-resistant epilepsy that patient-specific virtual brain models 
derived from diffusion magnetic resonance imaging have sufficient predictive 
power to improve diagnosis and surgery outcome. In partial epilepsy, seizures 
originate in a local network, the so-called epileptogenic zone, before 
recruiting other close or distant brain regions. We create personalized 
large-scale brain networks for 15 patients and simulate the individual seizure 
propagation patterns. Model validation is performed against the presurgical 
stereotactic electroencephalography data and the standard-of-care clinical 
evaluation. We demonstrate that the individual brain models account for the 
patient seizure propagation patterns, explain the variability in postsurgical 
success, but do not reliably augment with the use of patient-specific 
connectivity. Our results show that connectome-based brain network models have 
the capacity to explain changes in the organization of brain activity as 
observed in some brain disorders, thus opening up avenues towards discovery of 
novel clinical interventions.




5. Sci Rep. 2018 Dec 6;8(1):17687. doi: 10.1038/s41598-018-36047-2.

A Perlin Noise-Based Augmentation Strategy for Deep Learning with Small Data 
Samples of HRCT Images.

Deep learning is now widely used as an efficient tool for medical image 
classification and segmentation. However, conventional machine learning 
techniques are still more accurate than deep learning when only a small dataset 
is available. In this study, we present a general data augmentation strategy 
using Perlin noise, applying it to pixel-by-pixel image classification and 
quantification of various kinds of image patterns of diffuse interstitial lung 
disease (DILD). Using retrospectively obtained high-resolution computed 
tomography (HRCT) images from 106 patients, 100 regions-of-interest (ROIs) for 
each of six classes of image patterns (normal, ground-glass opacity, reticular 
opacity, honeycombing, emphysema, and consolidation) were selected for deep 
learning classification by experienced thoracic radiologists. For 
extra-validation, the deep learning quantification of the six classification 
patterns was evaluated for 92 HRCT whole lung images for which hand-labeled 
segmentation masks created by two experienced radiologists were available. 
FusionNet, a convolutional neural network (CNN), was used for training, test, 
and extra-validation on classifications of DILD image patterns. The accuracy of 
FusionNet with data augmentation using Perlin noise (89.5%, 49.8%, and 55.0% for 
ROI-based classification and whole lung quantifications by two radiologists, 
respectively) was significantly higher than that with conventional data 
augmentation (82.1%, 45.7%, and 49.9%, respectively). This data augmentation 
strategy using Perlin noise could be widely applied to deep learning studies for 
image classification and segmentation, especially in cases with relatively small 
datasets.




6. Brain Struct Funct. 2020 Jan;225(1):85-119. doi: 10.1007/s00429-019-01987-6. 
Epub 2019 Nov 26.

Mapping the human middle longitudinal fasciculus through a focused 
anatomo-imaging study: shifting the paradigm of its segmentation and 
connectivity pattern.

Τhe middle longitudinal fasciculus (MdLF) was initially identified in humans as 
a discrete subcortical pathway connecting the superior temporal gyrus (STG) to 
the angular gyrus (AG). Further anatomo-imaging studies, however, proposed more 
sophisticated but conflicting connectivity patterns and have created a vague 
perception on its functional anatomy. Our aim was, therefore, to investigate the 
ambiguous structural architecture of this tract through focused cadaveric 
dissections augmented by a tailored DTI protocol in healthy participants from 
the Human Connectome dataset. Three segments and connectivity patterns were 
consistently recorded: the MdLF-I, connecting the dorsolateral Temporal Pole 
(TP) and STG to the Superior Parietal Lobule/Precuneus, through the Heschl's 
gyrus; the MdLF-II, connecting the dorsolateral TP and the STG with the 
Parieto-occipital area through the posterior transverse gyri and the MdLF-III 
connecting the most anterior part of the TP to the posterior border of the 
occipital lobe through the AG. The lack of an established termination pattern to 
the AG and the fact that no significant leftward asymmetry is disclosed tend to 
shift the paradigm away from language function. Conversely, the theory of 
"where" and "what" auditory pathways, the essential relationship of the MdLF 
with the auditory cortex and the functional role of the cortical areas 
implicated in its connectivity tend to shift the paradigm towards auditory 
function. Allegedly, the MdLF-I and MdLF-II segments could underpin the 
perception of auditory representations; whereas, the MdLF-III could potentially 
subserve the integration of auditory and visual information.




7. Front Neuroanat. 2016 Sep 27;10:92. doi: 10.3389/fnana.2016.00092. eCollection 
2016.

An In vivo Multi-Modal Structural Template for Neonatal Piglets Using High 
Angular Resolution and Population-Based Whole-Brain Tractography.

An increasing number of applications use the postnatal piglet model in 
neuroimaging studies, however, these are based primarily on T1 weighted image 
templates. There is a growing need for a multimodal structural brain template 
for a comprehensive depiction of the piglet brain, particularly given the 
growing applications of diffusion weighted imaging for characterizing tissue 
microstructures and white matter organization. In this study, we present the 
first multimodal piglet structural brain template which includes a T1 weighted 
image with tissue segmentation probability maps, diffusion weighted metric 
templates with multiple diffusivity maps, and population-based whole-brain fiber 
tracts for postnatal piglets. These maps provide information about the integrity 
of white matter that is not available in T1 images alone. The availability of 
this diffusion weighted metric template will contribute to the structural 
imaging analysis of the postnatal piglet brain, especially models that are 
designed for the study of white matter diseases. Furthermore, the 
population-based whole-brain fiber tracts permit researchers to visualize the 
white matter connections in the piglet brain across subjects, guiding the 
delineation of a specific white matter region for structural analysis where 
current diffusion data is lacking. Researchers are able to augment the tracts by 
merging tracts from their own data to the population-based fiber tracts and thus 
improve the confidence of the population-wise fiber distribution.




8. Neuroimage Clin. 2016 May 19;11:707-718. doi: 10.1016/j.nicl.2016.05.010. 
eCollection 2016.

Whole-brain analytic measures of network communication reveal increased 
structure-function correlation in right temporal lobe epilepsy.

The in vivo structure-function relationship is key to understanding brain 
network reorganization due to pathologies. This relationship is likely to be 
particularly complex in brain network diseases such as temporal lobe epilepsy, 
in which disturbed large-scale systems are involved in both transient electrical 
events and long-lasting functional and structural impairments. Herein, we 
estimated this relationship by analyzing the correlation between structural 
connectivity and functional connectivity in terms of analytical network 
communication parameters. As such, we targeted the gradual topological 
structure-function reorganization caused by the pathology not only at the whole 
brain scale but also both in core and peripheral regions of the brain. We 
acquired diffusion (dMRI) and resting-state fMRI (rsfMRI) data in seven 
right-lateralized TLE (rTLE) patients and fourteen healthy controls and analyzed 
the structure-function relationship by using analytical network communication 
metrics derived from the structural connectome. In rTLE patients, we found a 
widespread hypercorrelated functional network. Network communication analysis 
revealed greater unspecific branching of the shortest path (search information) 
in the structural connectome and a higher global correlation between the 
structural and functional connectivity for the patient group. We also found 
evidence for a preserved structural rich-club in the patient group. In sum, 
global augmentation of structure-function correlation might be linked to a 
smaller functional repertoire in rTLE patients, while sparing the central core 
of the brain which may represent a pathway that facilitates the spread of 
seizures.




9. Comput Methods Programs Biomed. 2021 Apr 24;206:106113. doi: 
10.1016/j.cmpb.2021.106113. Online ahead of print.

Data augmentation using generative adversarial neural networks on brain 
structural connectivity in multiple sclerosis.

BACKGROUND AND OBJECTIVE: Machine learning frameworks have demonstrated their 
potentials in dealing with complex data structures, achieving remarkable results 
in many areas, including brain imaging. However, a large collection of data is 
needed to train these models. This is particularly challenging in the biomedical 
domain since, due to acquisition accessibility, costs and pathology related 
variability, available datasets are limited and usually imbalanced. To overcome 
this challenge, generative models can be used to generate new data.
METHODS: In this study, a framework based on generative adversarial network is 
proposed to create synthetic structural brain networks in Multiple Sclerosis 
(MS). The dataset consists of 29 relapsing-remitting and 19 
secondary-progressive MS patients. T1 and diffusion tensor imaging (DTI) 
acquisitions were used to obtain the structural brain network for each subject. 
Evaluation of the quality of newly generated brain networks is performed by (i) 
analysing their structural properties and (ii) studying their impact on 
classification performance.
RESULTS: We demonstrate that advanced generative models could be directly 
applied to the structural brain networks. We quantitatively and qualitatively 
show that newly generated data do not present significant differences compared 
to the real ones. In addition, augmenting the existing dataset with generated 
samples leads to an improvement of the classification performance (F1score 81%) 
with respect to the baseline approach (F1score 66%).
CONCLUSIONS: Our approach defines a new tool for biomedical application when 
connectome-based data augmentation is needed, providing a valid alternative to 
usual image-based data augmentation techniques.




10. Front Neuroinform. 2019 May 29;13:33. doi: 10.3389/fninf.2019.00033. eCollection 
2019.

Evaluation of Enhanced Learning Techniques for Segmenting Ischaemic Stroke 
Lesions in Brain Magnetic Resonance Perfusion Images Using a Convolutional 
Neural Network Scheme.

Magnetic resonance (MR) perfusion imaging non-invasively measures cerebral 
perfusion, which describes the blood's passage through the brain's vascular 
network. Therefore, it is widely used to assess cerebral ischaemia. 
Convolutional Neural Networks (CNN) constitute the state-of-the-art method in 
automatic pattern recognition and hence, in segmentation tasks. But none of the 
CNN architectures developed to date have achieved high accuracy when segmenting 
ischaemic stroke lesions, being the main reasons their heterogeneity in 
location, shape, size, image intensity and texture, especially in this imaging 
modality. We use a freely available CNN framework, developed for MR imaging 
lesion segmentation, as core algorithm to evaluate the impact of enhanced 
machine learning techniques, namely data augmentation, transfer learning and 
post-processing, in the segmentation of stroke lesions using the ISLES 2017 
dataset, which contains expert annotated diffusion-weighted perfusion and 
diffusion brain MRI of 43 stroke patients. Of all the techniques evaluated, data 
augmentation with binary closing achieved the best results, improving the mean 
Dice score in 17% over the baseline model. Consistent with previous works, 
better performance was obtained in the presence of large lesions.




11. Neuroimage. 2021 Jun;233:117934. doi: 10.1016/j.neuroimage.2021.117934. Epub 
2021 Mar 16.

Deep learning based segmentation of brain tissue from diffusion MRI.

Segmentation of brain tissue types from diffusion MRI (dMRI) is an important 
task, required for quantification of brain microstructure and for improving 
tractography. Current dMRI segmentation is mostly based on anatomical MRI (e.g., 
T1- and T2-weighted) segmentation that is registered to the dMRI space. However, 
such inter-modality registration is challenging due to more image distortions 
and lower image resolution in dMRI as compared with anatomical MRI. In this 
study, we present a deep learning method for diffusion MRI segmentation, which 
we refer to as DDSeg. Our proposed method learns tissue segmentation from 
high-quality imaging data from the Human Connectome Project (HCP), where 
registration of anatomical MRI to dMRI is more precise. The method is then able 
to predict a tissue segmentation directly from new dMRI data, including data 
collected with different acquisition protocols, without requiring anatomical 
data and inter-modality registration. We train a convolutional neural network 
(CNN) to learn a tissue segmentation model using a novel augmented target loss 
function designed to improve accuracy in regions of tissue boundary. To further 
improve accuracy, our method adds diffusion kurtosis imaging (DKI) parameters 
that characterize non-Gaussian water molecule diffusion to the conventional 
diffusion tensor imaging parameters. The DKI parameters are calculated from the 
recently proposed mean-kurtosis-curve method that corrects implausible DKI 
parameter values and provides additional features that discriminate between 
tissue types. We demonstrate high tissue segmentation accuracy on HCP data, and 
also when applying the HCP-trained model on dMRI data from other acquisitions 
with lower resolution and fewer gradient directions.




12. JAMA Neurol. 2013 Jul;70(7):845-51. doi: 10.1001/jamaneurol.2013.38.

Resting-state functional magnetic resonance imaging activity and connectivity 
and cognitive outcome in traumatic brain injury.

IMPORTANCE: The study of brain activity and connectivity at rest provides a 
unique opportunity for the investigation of the brain substrates of cognitive 
outcome after traumatic axonal injury. This knowledge may contribute to improve 
clinical management and rehabilitation programs.
OBJECTIVE: To study functional magnetic resonance imaging abnormalities in 
signal amplitude and brain connectivity at rest and their relationship to 
cognitive outcome in patients with chronic and severe traumatic axonal injury.
DESIGN: Observational study.
SETTING: University of Barcelona and Hospital Clinic de Barcelona, Barcelona, 
and Institut Guttmann-Neurorehabilitation Hospital, Badalona, Spain.
PARTICIPANTS: Twenty patients with traumatic brain injury (TBI) were studied, 
along with 17 matched healthy volunteers.
INTERVENTIONS: Resting-state functional magnetic resonance imaging and diffusion 
tensor imaging data were acquired. After exploring group differences in 
amplitude of low-frequency fluctuations (ALFF), we studied functional 
connectivity within the default mode network (DMN) by means of independent 
component analysis, followed by a dual regression approach and seed-based 
connectivity analyses. Finally, we performed probabilistic tractography between 
the frontal and posterior nodes of the DMN.
MAIN OUTCOMES AND MEASURES: Signal amplitude and functional connectivity during 
the resting state, tractography related to DMN, and the association between 
signal amplitudes and cognitive outcome.
RESULTS: Patients had greater ALFF in frontal regions, which was correlated with 
cognitive performance. Within the DMN, patients showed increased connectivity in 
the frontal lobes. Seed-based connectivity analyses revealed augmented 
connectivity within surrounding areas of the frontal and left parietal nodes of 
the DMN. Fractional anisotropy of the cingulate tract was correlated with 
increased connectivity of the frontal node of the DMN in patients with TBI.
CONCLUSIONS AND RELEVANCE: Increased ALFF is related to better cognitive 
performance in chronic TBI. The loss of structural connectivity produced by 
damage to the cingulum tract explained the compensatory increases in functional 
connectivity within the frontal node of the DMN.




13. PLoS Comput Biol. 2016 Aug 9;12(8):e1005025. doi: 10.1371/journal.pcbi.1005025. 
eCollection 2016 Aug.

Modeling of Large-Scale Functional Brain Networks Based on Structural 
Connectivity from DTI: Comparison with EEG Derived Phase Coupling Networks and 
Evaluation of Alternative Methods along the Modeling Path.

In this study, we investigate if phase-locking of fast oscillatory activity 
relies on the anatomical skeleton and if simple computational models informed by 
structural connectivity can help further to explain missing links in the 
structure-function relationship. We use diffusion tensor imaging data and alpha 
band-limited EEG signal recorded in a group of healthy individuals. Our results 
show that about 23.4% of the variance in empirical networks of resting-state 
functional connectivity is explained by the underlying white matter 
architecture. Simulating functional connectivity using a simple computational 
model based on the structural connectivity can increase the match to 45.4%. In a 
second step, we use our modeling framework to explore several technical 
alternatives along the modeling path. First, we find that an augmentation of 
homotopic connections in the structural connectivity matrix improves the link to 
functional connectivity while a correction for fiber distance slightly decreases 
the performance of the model. Second, a more complex computational model based 
on Kuramoto oscillators leads to a slight improvement of the model fit. Third, 
we show that the comparison of modeled and empirical functional connectivity at 
source level is much more specific for the underlying structural connectivity. 
However, different source reconstruction algorithms gave comparable results. Of 
note, as the fourth finding, the model fit was much better if zero-phase lag 
components were preserved in the empirical functional connectome, indicating a 
considerable amount of functionally relevant synchrony taking place with near 
zero or zero-phase lag. The combination of the best performing alternatives at 
each stage in the pipeline results in a model that explains 54.4% of the 
variance in the empirical EEG functional connectivity. Our study shows that 
large-scale brain circuits of fast neural network synchrony strongly rely upon 
the structural connectome and simple computational models of neural activity can 
explain missing links in the structure-function relationship.