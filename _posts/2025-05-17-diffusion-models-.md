---
layout: post
title: "Generative Diffusion Models: A Prelude"
description: "An introduction to diffusion probabilistic models and their application in point cloud registration tasks."
comments: true
tags: [diffusion-models, deep-learning, generative-models, point-cloud-registration]
---

Diffusion probabilistic models, or simply Diffusion models [1, 2, 3, 4] have emerged as a popular choice for generative tasks in the last couple of years. These developments have consequently inspired a plethora of their application in various fields, including image generation [5], object detection [6], and image segmentation [7], among others.

The core idea behind these models dates back to the 2015 work: "Deep Unsupervised Learning using Non-equilibrium Thermodynamics" [2], which represents data as a probability distribution. This decade-old work drew inspiration from non-equilibrium statistical physics/thermodynamics, particularly the diffusion process, where particles move from a direction of high concentration to a direction of low concentration. The work introduces the concept of generating data from noise.

To quote [2], the authors describe the idea as follows:

> "The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data."

<br />
![Diffusion Process Illustration]({{ site.url }}/assets/images/diffusion-process-diagram.png)
*Figure 1: An illustration of the Diffusion process. The process involves a forward process, which adds noise to the data via a noise scheduler (such as cosine schedule in this case) and a learned reverse process, which attempts to remove the noise added during the forward process incrementally.*
<br />

Even though the core idea of Diffusion models was conceptualized back in 2015, the Denoising Diffusion Probabilistic Model (DDPM) [1], developed in 2020, was the first to successfully demonstrate the generative capabilities of the Diffusion model by showcasing high-quality image generation. The work involves a forward (or diffusion) process and a reverse process.

Figure 1 presents the core idea behind Diffusion models. The forward process, as formulated in [1], gradually deteriorates the data, eventually leading to pure noise, and the Reverse process learns to denoise it iteratively in an attempt to reconstruct the data. The formulation from [1] has been briefly discussed in the following sections.

#### 1. Forward Process
The Forward process is also known as the Diffusion process, and it involves adding noise to the data iteratively. This is equivalent to the diffusion process in thermodynamics, since in the case of diffusion models, diffusion from data (higher structure and lower randomness) towards noise (lower structure and higher randomness) occurs. Thus, the goal of the Forward process is to transform $$x_0$$progressively, a data sample from the training data distribution,$$p_{data}$$, to Gaussian noise, $$x_T$$. This can be represented via a Markov chain: $$x_0 \rightarrow x_1 \rightarrow ... \rightarrow x_T$$, representing a sequence of noisy data samples. To note that given the data sample, $$x_0$$, noise is added gradually, corresponding to the time stamps: 1, 2, ..., T.

The Forward process can be represented as the distribution:

$$
f(x_t|x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I\right)
$$

Here,
* $$t$$ is the time step (goes from 1 up to T)
* $$x_0$$is a data sample from the training data distribution,$$p_{data}$$
* $$\beta_t$$ is the noise coefficient and is obtained via a schedule (a linear [1] or a cosine schedule [8])
* $$x_t$$ is the noisy sample at the step t

But instead of adding noise step-by-step, a closed form formula exists for $$x_t$$, i.e. for the distribution $$q(x_t|x_0)$$. As shown in [1]:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Here,
* $$\bar{\alpha}t = \prod{i=1}^{t} \alpha_i$$. It is the cumulative product of $$\alpha$$'s up until t
* $$\alpha_i = 1 - \beta_i$$; where $$\beta_i$$is obtained according to the variance schedule$$\beta_1, ..., \beta_T$$

Equation (2) allows for direct computation of noisy data at any given timestamp, without needing to obtain the noisy samples for in between steps.

#### 2. Reverse Process
The Reverse Diffusion, or simply the reverse process, learns to restore data from the noise. It aims to denoise noisy data. The reverse process can be thus represented by the distribution: $$r(x_{t-1}|x_t)$$, which is impossible to compute. A neural network thus learns to approximately predict a parameterised normal distribution, or in other words, learns to denoise the noisy data, i.e., $$x_T$$to$$x_0$$. This can be represented via a reverse Markov chain: $$x_T \rightarrow x_{T-1} \rightarrow ... \rightarrow x_0$$. This approximated (learned) distribution, $$p_\theta(x_{t-1}|x_t)$$ can be expressed as:

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t I)
$$

Here,
* $$\mu_\theta(x_t, t)$$ is the parameterized mean of the distribution

#### 3. Objective Function
The objective function, in theory, can be defined via the negative log likelihood of the denoised data, $$x_0$$, as:

$$
\text{Loss} = -\log(p_\theta(x_0))
$$

But the problem is that Reverse process is based on a Markov chain as explained above and thus, $$x_0$$depends on$$x_1, x_2, ..., x_T$$. These are not computable as we don't have $$r(x_{t-1}|x_t)$$distribution. To solve this problem, the authors of [1] derive a variational lower bound to indirectly optimise$$-\log(p_\theta(x_0))$$. To do so, they use the Kullback-Leibler (KL) divergence, a measure of how similar two distributions are. They obtain the KL divergence between $$f(x_{1:T}|x_0)$$, the desired (target) distribution, to which we have access, and $$p_\theta(x_{1:T}|x_0)$$, the approximated (learned) distribution. The terms are further simplified to eventually obtain the L2 norm as the optimising function, which needs to be minimised. The complete derivation for this is out of scope for this work and can be found in [1].

Once trained, the denoising module/reverse diffusion module is directly used to generate new data. To do so, a sample is taken from the Gaussian noise and is then iteratively denoised, leading to the generation of data from noise.



<br />
<hr />

#### References

[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.  
[2] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Non-equilibrium Thermodynamics. *ICML 2015*.  
[3] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS 2019*.  
[4] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations.  
[5] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.  
[6] Chen, S., Sun, P., Song, Y., & Luo, P. (2023). DiffusionDet: Diffusion Model for Object Detection. *ICCV 2023*.  
[7] Baranchuk, D., Rubachev, I., Voynov, A., Khrulkov, V., & Babenko, A. (2022). Label-Efficient Semantic Segmentation with Diffusion Models. *ICLR 2022*.  
[8] Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*.  
[9] Author Name. (Year). SE(3) Diffusion for Point Cloud Registration.  
[10] Author Name. (Year). Diffusion-based Point Cloud Registration.  
[11] Author Name. (Year). Correspondence Diffusion for Point Cloud Registration.