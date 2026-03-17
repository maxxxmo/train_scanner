# TRAIN_SCANNER
![Project Diagramm](images/project_diagramm.png)

The main goal is to create a real time system (Raspberry pi) that receive data and perform detection. Once the detection done the result is sent back to a computer.
The model used in the Raspberry is trained in a normal computer (no cloud needeed) and results are composed of different metrics and consumption analysis.


# Sources (not ordered)

- Ultralytics [Yolo](https://docs.ultralytics.com/fr/)

- [mlflow](https://mlflow.org/)

- [pytorch] (https://pytorch.org/)

- [ONNX](https://onnx.ai/)

# Data Sources

I will use FRSIGN dataset : [FRSIGNDATASET](https://frsign.irt-systemx.fr/)

Citation :
>@ARTICLE{2020arXiv200205665H,
       author = {{Harb}, Jeanine and {R{\'e}b{\'e}na}, Nicolas and {Chosidow}, Rapha{\"e}l and {Roblin}, Gr{\'e}goire and {Potarusov}, Roman and {Hajri}, Hatem},
        title = "{FRSign: A Large-Scale Traffic Light Dataset for Autonomous Trains}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computers and Society, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = "2020",
        month = "Feb",
          eid = {arXiv:2002.05665},
        pages = {arXiv:2002.05665},
archivePrefix = {arXiv},
       eprint = {2002.05665},
 primaryClass = {cs.CY},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200205665H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
>


