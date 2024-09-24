# HC-reg-seg

Fetus head circumference (HC) estimation from ultrasound images via segmentation-based and segmentation-free approaches. 
---
![Cover](code/hc.png)    
⭐ The highlight of this work:
* The HC is estimated via segmentation-based and segmentation-free approaches respectively;
* Methodology on evaluating the segmentation-based and segmentation-free methods;
* The evaluation of two approaches is performed under a **fair experimental environment**. 

👉 We evaluated both two types of approaches in several aspects:    
* The architectures (Memory, Parameters);     
* The prediction accuracy;
* Agreement analysis;
* Saliency maps;
* Actual inference time and memory cost;
* Comparison with state-of-the-art.

💻 About the code:    

The code is implemented with Python 3.* and the Deep learning library Tensorflow (Keras 2.*).

---
The work is finished together with [Caroline Petitjean](http://pagesperso.litislab.fr/cpetitjean/) and [Samia Ainouz](https://pagesperso.litislab.fr/sainouz/) in [LITIS](https://www.litislab.fr/) lab.    

Please consider citing this paper when you find it useful:

```
@inproceedings{zhang2020direct,
  title={Direct estimation of fetal head circumference from ultrasound images based on regression CNN},
  author={Zhang, Jing and Petitjean, Caroline and Lopez, Pierre and Ainouz, Samia},
  booktitle={Medical Imaging with Deep Learning},
  pages={914--922},
  year={2020},
  organization={PMLR}
}

@article{zhang2022segmentation,
  title={Segmentation-based vs. regression-based biomarker estimation: a case study of fetus head circumference assessment from ultrasound images},
  author={Zhang, Jing and Petitjean, Caroline and Ainouz, Samia},
  journal={Journal of Imaging},
  volume={8},
  number={2},
  pages={23},
  year={2022},
  publisher={MDPI}
}
```
