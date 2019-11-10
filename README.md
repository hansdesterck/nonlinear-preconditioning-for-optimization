# Nonlinear Preconditioning Project: Convergence Acceleration for Nonlinear Optimization

## Project goals:
* This project develops a collection of methods that can accelerate the convergence of simple fixed-point optimization methods (for example, Alternating Least Squares (ALS) for canonical tensor decomposition), *by using the fixed-point method as a* **Nonlinear Preconditioner** (**inner iteration**) *for well-known optimization methods (used as the **outer iteration**)* that include:
    * LBFGS
    * Nesterov's method
    * nonlinear Conjugate Gradients (NCG)
    * nonlinear GMRES (NGMRES)
    * Anderson acceleration (which is almost identical to NGMRES)
* In this approach, *the outer iteration method (LBFGS, Nesterov, NCG, NGMRES, Anderson) can be viewed as an accelerator for the inner iteration (e.g., ALS)*, or, equivalently, *the inner iteration (ALS) can be viewed as a Nonlinear Preconditioner for the outer iteration*

## Project contents/folders:
* **folder poblano_toolbox_ext:** contains **new implementations of the LBFGS, Nesterov, NCG and NGMRES/Anderson methods, that enable Nonlinear Preconditioning of these methods (e.g., using ALS as the inner iteration Nonlinear Preconditioner)**; in essence, these extensions replace the gradient directions that are normally used by the outer iteration methods, by preconditioner step directions, see references below; the new implementations are based on optimization methods implemented in the Poblano Toolbox for MATLAB (included in this repository), see next bullet point; *the implementations in folder poblano_toolbox_ext can be used as an extension of the Poblano Toolbox for MATLAB*
* **folder poblano_toolbox:** contains Poblano Toolbox for MATLAB (Sandia National Labs), Version 1.2 26-APR-2019, https://github.com/sandialabs/poblano_toolbox
* **folder poblano_precond_examples_tensor:** run **test_driver_CPtensor.m** to see the nonlinearly preconditioned methods in action, for *a canonical tensor decomposition **example** using ALS as nonlinear preconditioner*; this example makes use of the Tensor Toolbox for MATLAB (included in this repository), see next bullet point
* **folder tensor_toolbox-3.1:** contains Tensor Toolbox for MATLAB (Sandia National Labs), Version 3.1 04-Jun-2019, https://gitlab.com/tensors/tensor_toolbox
* **folder poblano_precond_examples_notensor:** contains a **further example** of using the nonlinearly preconditioned outer iterations, for a multidimensional generalization of Rosenbrock’s “banana” function with 100 variables; here, no problem-specific Nonlinear Preconditioner method is used, so we revert to the standard approach of using the steepest descent direction (negative gradient) as the preconditioning direction (this example does not use the Tensor Toolbox)

## References:

* De Sterck, H., 2012. **A nonlinear GMRES optimization algorithm for canonical tensor decomposition.** SIAM Journal on Scientific Computing, 34(3), pp.A1351-A1379.
* De Sterck, H., 2013. **Steepest descent preconditioning for nonlinear GMRES optimization.** Numerical Linear Algebra with Applications, 20(3), pp.453-471.
* De Sterck, H. and Winlaw, M., 2015. **A nonlinearly preconditioned conjugate gradient algorithm for rank‐R canonical tensor approximation.** Numerical Linear Algebra with Applications, 22(3), pp.410-432.
* Winlaw, M., Hynes, M.B., Caterini, A. and De Sterck, H., 2015. **Algorithmic acceleration of parallel ALS for collaborative filtering: Speeding up distributed big data recommendation in spark.** In 2015 IEEE 21st International Conference on Parallel and Distributed Systems (ICPADS) (pp. 682-691).
* De Sterck, H. and Howse, A., 2016. **Nonlinearly preconditioned optimization on grassmann manifolds for computing approximate tucker tensor decompositions.** SIAM Journal on Scientific Computing, 38(2), pp.A997-A1018.
* De Sterck, H. and Howse, A.J., 2018. **Nonlinearly preconditioned L‐BFGS as an acceleration mechanism for alternating least squares with application to tensor decomposition.** Numerical Linear Algebra with Applications, 25(6), p.e2202.
* Mitchell, D., Ye, N. and De Sterck, H., 2018. **Nesterov Acceleration of Alternating Least Squares for Canonical Tensor Decomposition: Momentum Step Size Selection and Restart Mechanisms.** arXiv preprint arXiv:1810.05846.
