# GPC-Glulam-Timber-Columns-Fire-Resistance

This repository contains the computational code accompanying the MSc thesis:

**"Improving Fire Resistance Prediction of Glulam Timber Columns Using Gaussian Process Surrogate Modelling"**

**Justus van der Wulp**  
MSc Civil Engineering – Structural Engineering  
Delft University of Technology (TU Delft), 2026  

The work combines numerical fire simulations and Gaussian Process (GPC) surrogate modelling to predict the burnout resistance of glulam timber columns under fire exposure.

---

## Repository Structure

### `SAFIRNumericalSimulations`
Input files and scripts related to the finite element fire simulations used. 

### `DefineBounds`
Scripts used to define parameter ranges for the simulation and modelling framework.

### `BigModel`
Implementation of the primary Gaussian Process surrogate model(s) used for burnout resistance prediction based on the numerical simulations.
Parts of the implementation in this folder is based on the methodology presented in:

Guang Zhao, Edward R. Dougherty, Byung-Jun Yoon, Francis J. Alexander, and Xiaoning Qian (2021).
Efficient Active Learning for Gaussian Process Classification by Error Reduction.
Proceedings of NeurIPS 2021.

### `ActiveLearningToy`
Simplified or exploratory implementation of the active learning strategy used during model development.
Parts of the implementation in this folder is based on the methodology presented in:

Guang Zhao, Edward R. Dougherty, Byung-Jun Yoon, Francis J. Alexander, and Xiaoning Qian (2021).
Efficient Active Learning for Gaussian Process Classification by Error Reduction.
Proceedings of NeurIPS 2021.

## Citation

If you use this code, please cite the associated MSc thesis.
