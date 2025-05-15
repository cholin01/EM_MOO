# Ab novo multi-objective generation framework for energetic materials with trading off energy and stability

This is the code for the paper "Ab novo multi-objective generation framework for energetic materials with trading off energy and stability". Energetic Materials (EMs) play important roles in military, civilian and aerospace fields. Energy and stability are two most important but contradictory properties in practical application, thus leading to a difficult challenge in developing new EMs with high comprehensive performance. Here, we propose an effective multi-objective generation framework by integrating a deep learning generator, machine learning prediction models, Pareto front optimization and quantum mechanics (QM) validation. Our design framework includes three main steps: (i) Molecule generation: RNN coupled with transfer learning is exploited to generate a new massive EMs search space. (ii) Properties prediction: development of ML models that accurately predict the relationship between molecular structure and Q and BDE values. (iii) Multi-objective optimization: Two-Dimensional Improved Probability-based (2D P[I]) Efficient Global Optimization method is exploited to select EMs with trading-off energy and stability based on the predicted values of ML models.

![Fig 1](https://github.com/user-attachments/assets/0448d946-ae57-485d-9855-cfc5e45c4f71)

## Guide for using the code

### Setup and Installation locally

1. Clone the repository to your local machine

2. It is recommended to use a virtual environment to avoid conflicts with other packages on your system. You could install required dependencies according```Q_model/README.md``` and ```RNN_generation/README.md```.

3. Run Python scripts directly

### Open in Colab

To ensure reproducibility across platforms, we have included a Google Colab notebook (```EM_colab.ipynb```). This makes it easy to run the code without needing to set up the environment locally. 

### Database

778 energetic molecules we collected with their Q and BDE values is provided.

Smiles: ```RNN_generation/data/emsmiles.csv```

BDE values: ```BDE_model/BDE_DATA.xlsx```

Q values: ```Q_model/data/778Q/Q_data.xlsx```


### Molecular generation

The folder RNN_generation contains the molecular generation model (RNN) and you can generate molecules with our fine-tuned model ```RNN_generation/record/gutl-0050-12.2942.pth``` by simply running

```
  python launcher_of_clm.py
```

detailed instruction for RNN's use is also provided.```RNN_generation/README.md```


### Prediction of BDE_values

The best BDE_prediction model in this work is supported by the folder ```BDE_model```, you can use by running

```
python BDE_model/predict.py
```

```BDE_model/descriptor.py``` is the code for calculating the descriptors


### Prediction of Q_values

The best Q_prediction model in this work is supported by the folder ```Q_model```, you can use by running 

```
python Q_model/script/finetuning_mdn.py
```
detailed instruction for using the model is also provided. ``` Q_model/README.md```

### Multi-objective optimization

for 2D P[I] Efficient Global Optimization method to select EMs with trading-off energy and stability, simply run

```python Multi_objective_optimization/MOO.py```
