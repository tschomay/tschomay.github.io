---
layout: page
title: Ted Schomay's Research
---

## Research
<p align="justify">
My research interests have sat at the intersection of math and computational biology. My PhD research involved developing mathematical frameworks to find patterns in big datasets. I used these methods to analyze genome-scale profiles of cancer patients in search of better prognostic tests. 
</p>

#### Mathematics
<p align="justify">
I mathematically defined and proved the properties of novel methods for the simultaneous decomposition of two or more data tensors (multi-way arrays). These new mathematical frameworks find prevalent patterns in multiple large and complicated datasets. These patterns illuminate the primary signals in the dataset and separate them from noise and batch effects (i.e., artifacts resulting from experiment or outside sources that affect a subset of the samples).
</p>

<p align="justify">
These methods build on the singular value decomposition (SVD) and generalized SVD. Given a dataset of two or more tensors that are dimension-matched on all axes except for one, these methods allow us to find sets of basis vectors (or stereotypical patterns across each dimension) that are shared among the matched dimensions of the tensors. These basis vectors contain fundamental information about the dataset. With the overall structure of the decomposition, we can interpret patterns that are common between all the tensors in the decomposition or unique to only some. We can also use the generalizations of the singular values in these decompositions to determine the importance of each pattern in the dataset. The advantage over previous methods is the ability to include higher dimensional data, enabling modeling and controlling for more variables.
</p>


#### Computational Biology
<p align="justify">
DNA copy-number aberrations (CNAs) are a well-known hallmark of cancer. However, their exact function is not well understood. I have used the mathematical decompositions described above to uncover patterns of CNAs that predict patient prognosis. These genomic-based prognostic indicators could influence patient treatment and potentially contain insight into the progression of the disease.
</p>

<p align="justify">
The new unsupervised mathematical frameworks enabled us to find patterns of CNAs in ovarian cancer that are exclusive to tumor tissue compared with normal, are independent of the microarray used to measure the data (platform bias), and separate the patient set into two groups with significantly different survival times. This resulted in a genomic-based prognostic indicator for ovarian cancer that outperforms and is independent of the current clinical indicators.
</p>

### Links
[Home](/) | [Research](/research) | [About Me](/about) | [Blog](/blog)
