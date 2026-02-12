

Diffusion probabilistic models, or simply Diffusion models [[1](#ref1), [2](#ref2), [3](#ref3), [4](#ref4)] have emerged as a popular choice for generative tasks in the last couple of years. These developments have consequently inspired a plethora of their application in various fields, including image generation [[5](#ref5)], object detection [[6](#ref6)], and image segmentation [[7](#ref7)], among others.

The core idea behind these models dates back to the 2015 work: "Deep Unsupervised Learning using Non-equilibrium Thermodynamics" [[2](#ref2)], which represents data as a probability distribution. This decade-old work drew inspiration from non-equilibrium statistical physics/thermodynamics, particularly the diffusion process, where particles move from a direction of high concentration to a direction of low concentration. The work introduces the concept of generating data from noise.

To quote [[2](#ref2)], the authors describe the idea as follows:

> "The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data."

<div class="figure">
  <img src="/assets/images/diffusion-process-diagram.png" alt="Diffusion Process Illustration">
  <p class="caption"><strong>Figure 1:</strong> An illustration of the Diffusion process. The process involves a forward process, which adds noise to the data via a noise scheduler (such as cosine schedule in this case) and a learned reverse process, which attempts to remove the noise added during the forward process incrementally.</p>
</div>

Even though the core idea of Diffusion models was conceptualized back in 2015, the Denoising Diffusion Probabilistic Model (DDPM) [[1](#ref1)], developed in 2020, was the first to successfully demonstrate the generative capabilities of the Diffusion model by showcasing high-quality image generation. The work involves a forward (or diffusion) process and a reverse process.

Figure 1 presents the core idea behind Diffusion models. The forward process, as formulated in [[1](#ref1)], gradually deteriorates the data, eventually leading to pure noise, and the Reverse process learns to denoise it iteratively in an attempt to reconstruct the data. The formulation from [[1](#ref1)] has been briefly discussed in the following sections.

## Forward Process

The Forward process is also known as the Diffusion process, and it involves adding noise to the data iteratively. This is equivalent to the diffusion process in thermodynamics, since in the case of diffusion models, diffusion from data (higher structure and lower randomness) towards noise (lower structure and higher randomness) occurs. Thus, the goal of the Forward process is to transform $x_0$ progressively, a data sample from the training data distribution, $p_{data}$, to Gaussian noise, $x_T$. This can be represented via a Markov chain: $x_0 \rightarrow x_1 \rightarrow ... \rightarrow x_T$, representing a sequence of noisy data samples. To note that given the data sample, $x_0$, noise is added gradually, corresponding to the time stamps: 1, 2, ..., T.

The Forward process can be represented as the distribution:

$$f(x_t|x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I\right)$$

<div class="equation-note">Equation (1)</div>

Here,

- $t$ is the time step (goes from 1 up to T)
- $x_0$ is a data sample from the training data distribution, $p_{data}$
- $\beta_t$ is the noise coefficient and is obtained via a schedule (a linear [[1](#ref1)] or a cosine schedule [[8](#ref8)])
- $x_t$ is the noisy sample at the step t

But instead of adding noise step-by-step, a closed form formula exists for $x_t$, i.e. for the distribution $q(x_t|x_0)$. As shown in [[1](#ref1)]:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

<div class="equation-note">Equation (2)</div>

Here,

- $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$. It is the cumulative product of $\alpha$'s up until t
- $\alpha_i = 1 - \beta_i$; where $\beta_i$ is obtained according to the variance schedule $\beta_1, ..., \beta_T$

Equation 2 allows for direct computation of noisy data at any given timestamp, without needing to obtain the noisy samples for in between steps.

## Reverse Process

The Reverse Diffusion, or simply the reverse process, learns to restore data from the noise. It aims to denoise noisy data. The reverse process can be thus represented by the distribution: $r(x_{t-1}|x_t)$, which is impossible to compute. A neural network thus learns to approximately predict a parameterised normal distribution, or in other words, learns to denoise the noisy data, i.e., $x_T$ to $x_0$. This can be represented via a reverse Markov chain: $x_T \rightarrow x_{T-1} \rightarrow ... \rightarrow x_0$. This approximated (learned) distribution, $p_\theta(x_{t-1}|x_t)$ can be expressed as:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t I)$$

<div class="equation-note">Equation (3)</div>

Here,

- $\mu_\theta(x_t, t)$ is the parameterized mean of the distribution

## Objective Function

The objective function, in theory, can be defined via the negative log likelihood of the denoised data, $x_0$, as:

$$\text{Loss} = -\log(p_\theta(x_0))$$

<div class="equation-note">Equation (4)</div>

But the problem is that Reverse process is based on a Markov chain as explained above and thus, $x_0$ depends on $x_1, x_2, ..., x_T$. These are not computable as we don't have $r(x_{t-1}|x_t)$ distribution. To solve this problem, the authors of [[1](#ref1)] derive a variational lower bound to indirectly optimise $-\log(p_\theta(x_0))$. To do so, they use the Kullback-Leibler (KL) divergence, a measure of how similar two distributions are. They obtain the KL divergence between $f(x_{1:T}|x_0)$, the desired (target) distribution, to which we have access, and $p_\theta(x_{1:T}|x_0)$, the approximated (learned) distribution. The terms are further simplified to eventually obtain the L2 norm as the optimising function, which needs to be minimised. The complete derivation for this is out of scope for this work and can be found in [[1](#ref1)].

Once trained, the denoising module/reverse diffusion module is directly used to generate new data. To do so, a sample is taken from the Gaussian noise and is then iteratively denoised, leading to the generation of data from noise.

## Diffusion in Point Cloud Registration

The success of diffusion models has inspired several works in the field of Point Cloud Registration (PCR). These works formulate PCR as a denoising diffusion process, either by attempting to generate a correct transformation from a noisy transformation [[9](#ref9), [10](#ref10)] or by attempting to obtain correct correspondence from a noisy correspondence matrix [[11](#ref11)]. Similar to how DDPM [[1](#ref1)] introduces a Forward process (data to noise) and a Reverse process (noise to data), these works introduce frameworks that work via a Forward and a Reverse process. The authors of [[9](#ref9), [11](#ref11)] suggest that the diffusion process acts as a natural data augmenter and helps create diversity in the training set, thus leading to far better generalizability.

For instance, [[9](#ref9)] introduces a diffusion-inspired PCR model, with the SE(3) Diffusion process. This differs from the standard diffusion processes (such as [[1](#ref1)]), which...

---

## References

<div class="references">

<p id="ref1">[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. <em>NeurIPS 2020</em>. <a href="PLACEHOLDER_LINK_1" target="_blank">Link</a></p>

<p id="ref2">[2] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Non-equilibrium Thermodynamics. <em>ICML 2015</em>. <a href="PLACEHOLDER_LINK_2" target="_blank">Link</a></p>

<p id="ref3">[3] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. <em>NeurIPS 2019</em>. <a href="PLACEHOLDER_LINK_3" target="_blank">Link</a></p>

<p id="ref4">[4] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. <a href="PLACEHOLDER_LINK_4" target="_blank">Link</a></p>

<p id="ref5">[5] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. <em>CVPR 2022</em>. <a href="PLACEHOLDER_LINK_5" target="_blank">Link</a></p>

<p id="ref6">[6] Chen, S., Sun, P., Song, Y., & Luo, P. (2023). DiffusionDet: Diffusion Model for Object Detection. <em>ICCV 2023</em>. <a href="PLACEHOLDER_LINK_6" target="_blank">Link</a></p>

<p id="ref7">[7] Baranchuk, D., Rubachev, I., Voynov, A., Khrulkov, V., & Babenko, A. (2022). Label-Efficient Semantic Segmentation with Diffusion Models. <em>ICLR 2022</em>. <a href="PLACEHOLDER_LINK_7" target="_blank">Link</a></p>

<p id="ref8">[8] Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. <em>ICML 2021</em>. <a href="PLACEHOLDER_LINK_8" target="_blank">Link</a></p>

<p id="ref9">[9] Author Name. (Year). SE(3) Diffusion for Point Cloud Registration. <a href="PLACEHOLDER_LINK_9" target="_blank">Link</a></p>

<p id="ref10">[10] Author Name. (Year). Diffusion-based Point Cloud Registration. <a href="PLACEHOLDER_LINK_10" target="_blank">Link</a></p>

<p id="ref11">[11] Author Name. (Year). Correspondence Diffusion for Point Cloud Registration. <a href="PLACEHOLDER_LINK_11" target="_blank">Link</a></p>

</div>