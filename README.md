# MOF-Methane-PINN
Insights into Methane Adsorption in Metal-organic Frameworks through Physics-informed Machine Learning
# Physics-Informed Neural Network (PINN) for Methane Adsorption Prediction
This project presents a Physics-Informed Neural Network (PINN) framework for predicting methane adsorption in Metal–Organic Frameworks (MOFs). The model combines a traditional Feedforward Neural Network (FNN) with physics-based monotonicity constraints to enhance interpretability, robustness, and physical consistency.  
The framework takes structural and geometric descriptors of MOFs (e.g., pore size, accessible volume, void fraction) as inputs and predicts methane adsorption capacity under given thermodynamic conditions.
# Data Generation
The dataset is derived from simulated methane adsorption isotherms combined with MOF structural features extracted from:  
CoRE-MOF and related databases  
Zeo++ calculations (e.g., pore limiting diameter, largest cavity diameter, accessible surface area)  
To improve generalization, additional synthetic datasets were generated within physically meaningful ranges of pore size and void fraction. These datasets include monotonicity factors to enforce physical constraints.
# Outputs & Evaluation
The framework provides:  
Predicted methane adsorption capacities across different MOFs  
Physically consistent adsorption trends with respect to structural descriptors  
Quantitative evaluation metrics including MAE, RMSE, and R²  
The results demonstrate that PINN models achieve better robustness against outliers and preserve physical trends compared to baseline FNN models.
# Data Access
Example dataset (Data.xlsx) is included in this repository.  
Full datasets can be provided upon request.
#  Code Structure & Execution Guide
Code was implemented in Python 3.9 with PyTorch, using Adam optimizer, hidden layers [128, 64, 32], learning rate = 0.001, batch size = 32, epochs = 80, and physics loss weight λp = 1.
