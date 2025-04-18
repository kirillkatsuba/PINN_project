# Approximation of Subsurface Flow via Physics Informed Neural Netwokrs
## Key idea of the research
Surrogate models of complicated physical systems are routinely utilized in engineering problems. For instance,
Uncertainty Quantification in hydrocarbons production or subsurface CO2 storage has an extremely high
computational cost that can be reduced dramatically by utilization of cheap proxy models. Huge variety of Machine-
Learning models have been iplemented to develop a proxy model starting from Polynomial functions and Decision
Trees and ending by Artificial Neural Networks.
We show experimentally that accurate Neural Network can be trained on relatively small dataset. Moreover, we
demonstrate that such Physics Informed Neural Networks last longer in comparision with such methods as
Polynomial Chaos Expansion and Decision Trees when the complexity of the systems increases and if trained on
the same dataset. In other words Physics Informed Neural Networks can approximate systems with high degree of
nonlinearity even when limited number of samples is available.

## Dataset
We have simulator that gives approximation of the reservoir. To build system of oil-water displacement we need
define porosity and permeability. And as a result we get meshes of velocities, saturations and pressures
depending on time. Here, some examples of simulations.


For the model training time was sampling on segment [0, 3], the coordinates if the surface belongs to [0, 1] x [0,
1].


We generate point in two dimensional surface, but we will add nonlinearity to the system via input parameters of
reservoir. This parameters will define permeability field, and therefore model should train its parameters to define
which system is considered.


## Results
Boosting and PCE can approximate systems with a low degree of nonlinearity quite well (1). However, if we
increase the system’s nonlinearity (as shown in the second test case by adding a nonlinear permeability field),
these algorithms can only achieve good accuracy with a large amount of data. For example, with 500 training
points, neither Boosting nor PCE were able to approximate the fluid propagation velocity in the reservoir, whereas
PINNs managed to do so with much higher accuracy.