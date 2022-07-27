# Data Augmentation for Time Series Classification with Deep Learning models

This is the companion repository for our paper titled "Data Augmentation for Time Series Classification with Deep Learning models". 
This paper has been accepted in the workshop [AALTD22@ECML22](https://project.inria.fr/aaltd22/) (Spt 19-23, 2022).

## Datasets used

All our experiments were performed on the widely used time series classification benchmark [UCR/UEA archive](http://timeseriesclassification.com/index.php). 
We used the latest version (2018) which contains 128 datasets. The UCR archive is necessary to run the code.

## Models used

All experiments were performed using the time series classifier InceptionTime from the paper [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939). 
The companion Github repository is located [here](https://github.com/hfawaz/InceptionTime). InceptionTime's weight are necessary to run the code.

## Requirements

The code runs using Python 3.7. You will need to install the packages present in the [requirements.txt](requirements.txt) file.

``pip install -r requirements.txt``

## Code

The code is divided as follows:

* The [main.py](main.py) python file contains the necessary code to run the experiments.

* The [models](models/) folder contains the implementation of Inception and InceptionTime.
* The [data_aug](data_aug/) folder contains the implementation of the data augmentation methods.
* The [data](data/) folder contains everything needed to load the UCR archive.
* The [utils](utils/) folder contains constants used troughout the code.

### Adaptions required

You should consider changing the [constants](utils/constants.py) file.
It contains the path of the UCR Archive and the folder where the outputs are generated.

## Reference

This work is not published yet.

## Acknowledgments
This work was funded by ArtIC project ”Artificial Intelligence for Care” (grant
ANR-20-THIA-0006-01) and co-funded by Région Grand Est, Inria Nancy -
Grand Est, IHU of Strasbourg, University of Strasbourg and University of Haute-
Alsace. 

The authors would like to thank the providers of the UCR archive as
well as the Mésocentre of Strasbourg for providing access to the GPU cluster.
