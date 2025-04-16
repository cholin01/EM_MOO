## Ab novo multi-objective generation framework for energetic materials with trading off energy and stability

This is the code for the paper "Ab novo multi-objective generation framework for energetic materials with trading off energy and stability". Energetic Materials (EMs) play important roles in military, civilian and aerospace fields. Energy and stability are two most important but contradictory properties in practical application, thus leading to a difficult challenge in developing new EMs with high comprehensive performance. Here, we propose an effective multi-objective generation framework by integrating a deep learning generator, machine learning prediction models, Pareto front optimization and quantum mechanics (QM) validation. Our design framework includes three main steps: (i) Molecule generation: RNN coupled with transfer learning is exploited to generate a new massive EMs search space. (ii) Properties prediction: development of ML models that accurately predict the relationship between molecular structure and Q and BDE values. (iii) Multi-objective optimization: Two-Dimensional Improved Probability-based (2D P[I]) Efficient Global Optimization method is exploited to select EMs with trading-off energy and stability based on the predicted values of ML models.

![Fig 1](https://github.com/user-attachments/assets/0448d946-ae57-485d-9855-cfc5e45c4f71)

### Guide for using the code

The database is saved in the folder Data, including the 778 energetic molecules with their Q and BDE values. 

The folder RNN_generation contains the molecular generation model (RNN) and instruction for its use.

The best models for the prediction of Q and BDE are supported by the folders “Q_model” and “BDE_model”. For the prediction of BDE values, ```descriptor.py``` is the code for calculating the SEC descriptors and CBD descriptors, and the feature of our 778 molecules is saved in this folder, named ```dataset.xlsx```. ```Predict.ipynb``` is provided if you wish to use our best model for predictiong the BDE of energectic materials. For the prediction of Q values, the best model, named 3D-GNN is supplied in the folder Q_model. 

```Multi_objective_optimization.ipynb``` is the code for 2D P[I] Efficient Global Optimization method to select EMs with trading-off energy and stability.
