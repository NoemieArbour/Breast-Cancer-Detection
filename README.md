# Breast-Cancer-Detection
This algorithm builds a Machine Learning model that detects breast cancer using logistic regression.
Made by Noémie Arbour

## Basic information
This project uses the Breast Cancer Wisconsin's dataset :

Wolberg,WIlliam. (1992). Breast Cancer Wisconsin (Original). 
UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z.

###Attribute Information:

1. Sample code number:            id number
2. Clump Thickness:               1 - 10
3. Uniformity of Cell Size:       1 - 10
4. Uniformity of Cell Shape:      1 - 10
5. Marginal Adhesion:             1 - 10
6. Single Epithelial Cell Size:   1 - 10
7. Bare Nuclei:                   1 - 10
8. Bland Chromatin:               1 - 10
9. Normal Nucleoli:               1 - 10
10. Mitoses:                      1 - 10
11. Class:                        (2 for benign, 4 for malignant)


###Class distribution:
357 benign, 212 malignant

## The model after 500 Epoch
0.01916491430879707(Clump Thickness) 
+ 0.37425648415565776(Uniformity of Cell Size) 
+ 0.21874139774741086(Uniformity of Cell Shape) 
+ 0.0955157344456059(Marginal Adhesion) 
+ -0.23664342873248626(Single Epithelial Cell Size) 
+ 0.38071240214538704(Bare Nuclei) 
+ -0.21721946447576623(Bland Chromatin) 
+ 0.215041519143409(Normal Nucleoli) 
+ 0.005511260639835064(Mitoses) 
+ -2.895198671336623

###Success rate:
100.0% success rate with the 50% standard threshold.

Detection rate with a 33.0% threshold : 100.0%

Within those, 0 were benign (False positives).