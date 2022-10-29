---
layout: post
title: Introduction to Boltzmann Machine
tags: My Blog
math: true
date: 2022-10-30 12:00 +0800
---

# Boltzmann Machine

## Introduction to BM

At the condition of thermal equilibrium,the relation of P(X),where X is a state,and temperature obey the law of Boltzmann distribution:

$$
P(X)\propto e^{-\frac{E(X)}{KT}}
$$

The toplogical structure of BM(Boltzmann Machine) is between the Hopfield Networks and BP Networks.

![Boltzmann Machine](/image/bolztmann_machine/BM.png)

In form,the BM Network is similar to DHNN,requiring \\\(w_{ij}=w_{ji}\\\) and \\\(w_{ii}=\\\)$.But as to the function,BM is more similar to BP Network with 3 layers(input,hidden and output).

The input of one neuron in BM can be written as:

$$
net_j=\sum_{i}^{n}(w_{ij}x_i-T_j)
$$ 
Different to DHNN,the output is determined by a transition probability,rather than being directly calculated by the signal function:
$$
P_j(1)=\frac{1}{1+e^{-net_j/T}}
$$
which is for the probablity of the output being 1,and also:
$$
P_j(0)=1-P_j(1)
$$

The energy function of BM is the same as the Hopfield Networks:

$$
\begin{aligned}
    E(t)&=-\frac{1}{2}X^T(t)WX(t)+X^T(t)T\\
    &=-\frac{1}{2}\sum_{j=1}^{n}\sum_{i=1}^{n}w_{ij}x_i x_j+\sum_{i=1}^{n}T_i x_i
\end{aligned}
$$ 
Assuming that the net works asynchronously,
$$
\Delta E(t)=-\Delta x_j(t)net_j(t)
$$

Due to the different way of gernerating output,the energy of the system could get higher in some cases which gives BM the ability to reach the global minimum.

- \\\(x_j:1 \to 0\\\):\\\(\Delta x=-1\\\),

  $$
  E_0-E_1=\Delta E=net_j \\
  \therefore P_j(1)=\frac{1}{1+e^{-\frac{\Delta E}{T}}} \\
  P_j(0)=1-P_j(1)=\frac{e^{-\frac{\Delta E}{T}}}{1+e^{\frac{-\Delta E}{T}}} \\
  \frac{P_j(0)}{P_j(1)}=e^{-\frac{\Delta E}{T}}=\frac{e^{-\frac{E_0}{T}}}{e^{-\frac{E_1}{T}}}
  $$
- \\\(x_j:0 \to 1\\\):\\\(\Delta x=1\\\),

  $$
  E_0-E_1=\Delta E=-net_j \\
  P_j(1)=\frac{1}{1+e^{-\frac{\Delta E}{T}}} \\
  P_j(0)=1-P_j(1)=\frac{e^{-\frac{\Delta E}{T}}}{1+e^{\frac{-\Delta E}{T}}} \\
  \frac{P_j(0)}{P_j(1)}=e^{-\frac{\Delta E}{T}}=\frac{e^{-\frac{E_0}{T}}}{e^{-\frac{E_1}{T}}}
  $$

Extending the formula to any two states of the net,we have

$$
\frac{P(\alpha)}{P(\beta)}=\frac{e^{-\frac{E_\alpha}{T}}}{e^{-\frac{E_\beta}{T}}}
$$

which is the famous Boltzmann Distribution.

## The training algorithm for BM

By supervised learning,BM Networks could simulate the probability distribution of different patterns of the trainng set,thus realizing associatice memory.The goal of learning is to fit the probability distribution by adjusting the weights in the network.

Assuming that there are \\\(P\\\) patterns in the training set,the number of hidden nodes in BM is \\\(m\\\) and of the visable nodes is \\\(n\\\)(usually \\\(P<n\\\)),the probability distribution of the training set can be written as:

$$
P(X^1),P(X^2,\cdots,P(X^P)
$$

When the training process is over,the probability of corresponding patterns is:

$$
P'(X^1),P'(X^2),\cdots ,P'(X^P)
$$

And we want the two distributions to be as similar as possible.

### The algorithm to achieve thermal equilibrium

1. Restrict the net by some rules,which we will see below
2. Randomly choose a free node \\\(j\\\) and update the state

   $$
   s_j(t+1)=\begin{cases}
           1,&s_j(t)=0 \\
           0,&s_j(t)=1
       \end{cases}
   $$
3. Calculate the change of energy:\\\(\Delta E_j=-\Delta s_j(t)net_j(t)\\\)
4. - \\\(\Delta E_j<0\\\),we accept the change;
   - \\\(\Delta E_j>0\\\):if \\\(P(s_j(t+1))>\rho\\\),we accept the change,or we maintain the original state;(\\\(\rho \in (0,1)\\\) is a known constant);
5. Go back to 2-4 until all the nodes have been chosen;
6. Low the temperature by a cooling way,like

   $$
   T(t)=\frac{T_0}{\log(1+t)}
   $$

   or

   $$
   T(t)=\frac{T_0}{1+t}
   $$
7. Go back to 2-6 until for all the nodes we have \\\(\Delta E_j=0\\\)(the system is heat balanced).

### The algorithm to adjust weights

Different weights lead to different distributions.We can use Kullback-Leibler Divergence to quantify the difference of distributions(\\\(P(X^i)\\\) and \\\(P'(X_i)\\\)) and the goal is to minimize it:

$$
\textbf{minimize}\quad G=\sum_{i}P(X^i)\ln \frac{P(X^i)}{P'(X^i)}
$$

Kullback-Leibler Divergence is used to quantify the \"distance\" between two distribution functions:

$$
\mathrm{KL}[P(X)||Q(X)]=\sum_{x \in X}[P(x\log \frac{P(x)}{Q(x)}]=E_{x \sim P(x)}[\log \frac{P(x)}{Q(x)}]
$$

If \\\(P(X)\\\) is the true distribution and \\\(Q(X)\\\) is the approximate distribution,we could adjust \\\(Q(X)\\\) to minimize KL,which means \\\(Q(X)\\\) is close to \\\(P(X)\\\).We should notice that
\\\(\mathrm{KL}[P||Q]\neq \mathrm{KL}[Q||P]\\\),thefirst one is called
forward Kullback-Leibler Divergence and the second one is reverse Kullback-Leibler Divergence.


\\\(P'(X^i)\\\) is determined by \\\(w_{ij}\\\),thus we need to calculate\\\(\frac{\partial G}{\partial w_{ij}}\\\).

$$
\frac{\partial G}{\partial w_{ij}}=-\frac{1}{T} (p_{ij}=p'_{ij})
$$

$$
\Delta w_{ij}=\varepsilon (p_{ij}-p'_{ij})
$$

There are two stages of the training process.The first one is the positive stage,when the state of visable units is restricted to a fixed dipolar vector, sampling from distribution \\\(P(X^i)\\\),then the hidden units are adjusted until the system is balanced.The second stage is called the negative stage,when all the units envolve freely to stable state.

- \\\(p_{ij}\\\):the possibility of both \\\(i\\\) and \\\(j\\\) neuron being turned on
  when the system is stable at the positive stage;
- \\\(p'_{ij}\\\):the possibility of both \\\(i\\\) and \\\(j\\\) neuron being turned on
  when the system is stable at the negative stage;

Algorithm:

1. Randomly initialized the weights of the net(\\\(w_{ij}(0)\\\));
2. At the positive stage,restrict the input to \\\(P(X^1),P(X^2),\cdots,P(X^P)\\\) according to \\\(P(X^i)\\\() and envolve as section 2.1 describes,then count \\\(p_{ij},\forall i,j\\\);
3. At the negative stage,the net envolves freely and count \\\(p'_{ij}\\\);
4. Update the weights as

$$
   \Delta w_{ij}=\eta(p'_{ij}-p_{ij});
$$

1. Repeat the above steps until \\\(p'_{ij}\\\) and \\\(p_{ij}\\\) get close.

### The running algorithm

1. Initialize the net:\\\(w_{ij},T_0,T_{end},\mathrm{input}\\\);
2. At \\\(T(n)\\\),choose a neuron to update:if the energy goes down,accept the change;else calculate \\\(P=\frac{1}{1+e^{-\Delta E/T(n)}}\\\) and compare it with a random number \\\(\varepsilon \in (0,1)\\\),if \\\(\varepsilon<P\\\( then we can accept the change.
3. Check whether the system is balanced at this temperature,if yes we continue to step 4,else we go to 2 to choose another neuron;
4. Cool the system and check whether \\\(T(n)=T_{end}\\\),if yes the process ends,else we go to step 2.
