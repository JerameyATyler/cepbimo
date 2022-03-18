
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JerameyATyler/mybinder_environment/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FJerameyATyler%252Fcepbimo%26urlpath%3Dlab%252Ftree%252Fcepbimo%252Fcepbimo%252Findex.ipynb%26branch%3Dmain) Click on the badge to view on MyBinder.
# Abstract
> A new model architecture is presented to predict room acoustical parameters from a running binaural signal. For this purpose, a deep neural network architecture is combined with a precedence effect model to extract the 
spatial and temporal locations of the direct signal and early reflections. The precedence effect model builds on the modified BICAM algorithm (Braasch, 2016), for which the 1st layer auto-/cross correlation functions are 
replaced with a Cepstrum method. The latter allows a better separation of features relating to the source signal's early reflections and harmonic structure. The precedence effect model is used to create binaural activity maps 
that are analyzed by the neural network for pattern recognition. Anechoic orchestral recordings were reverberated by adding four early reflections and late reverberation to test the model. Head-related transfer functions were 
used to spatialize direct sound and early reflections. The model can identify the main reflection characteristics of a room, offering applications in numerous fields, including room acoustical assessment, acoustical analysis 
for virtual-reality applications, and modeling of human perception.

## Data
> * Train set https://reflections.speakeasy.services/train.zip
> * Test set https://reflections.speakeasy.services/test.zip
> * Validate set https://reflections.speakeasy.services/validate.zip
> * Full set https://reflections.speakeasy.services/full.zip