Understood.
**No emojis**, and **no explicit figure placement instructions** — I will describe figures naturally in the text.

Below is the **full long-form, ~10-page, ~5000+ word Medium-style article** explaining
**“Mechanisms of Projective Composition of Diffusion Models”**
in clear, simple language while maintaining full technical accuracy.

---

# **From Noise to Structure: A Deep, Simple Explanation of “Mechanisms of Projective Composition of Diffusion Models”**

*By a senior ML researcher writing for motivated beginners, engineers, and curious readers.*

---

## **Introduction**

Diffusion models have already changed how we think about image generation, creativity, and controllability. Over the last few years, these models have gained the remarkable ability to combine concepts: a dog in the style of Monet, a chair made of strawberries, or a landscape with multiple objects placed exactly where you want.

But there has always been a deeper question behind these magical combinations:

**How do diffusion models combine concepts at all?**
**Why do some concepts combine naturally while others fail completely?**
**What makes “composition” possible, and when should we expect it to fail?**

The paper *Mechanisms of Projective Composition of Diffusion Models* tackles these questions head-on. It provides the first mathematically complete framework explaining when diffusion models can compose different pretrained distributions, why simple score addition sometimes works, why it sometimes collapses, and how to predict success or failure in advance.

More importantly, the authors uncover a unifying mechanism behind composition:
**projective composition**, a definition that retains only the parts of each distribution that matter and allows the generated output to be genuinely out-of-distribution relative to the original models.

This article explains the entire paper in simple language, carefully unpacking the definitions, equations, intuitions, and experimental insights. My goal is to give you a researcher’s mental model—something deeper than a summary, but approachable enough that you can carry these ideas into your own work.

---

# **1. Background: What Is Composing Generative Models?**

Before diving into the paper’s contributions, let’s start from the broader problem.

### **1.1 Generative models try to model a distribution over data**

A generative model learns a probability distribution:

[
p_{\theta}(x)
]

that captures the variability and structure of images, text, or audio.

In diffusion models, we do this by progressively adding noise to real data, learning to reverse this process, and sampling from a clean distribution by denoising from pure noise.

### **1.2 Composition means combining multiple distributions**

Suppose we have:

* (p_{\text{dog}}): distribution of dog photos
* (p_{\text{oil}}): distribution of oil-painting style images
* (p_{u}): unconditional distribution of generic images

The classic trick (used in many prior works) is:

[
\nabla_x \log p_{\text{dog}} +
\nabla_x \log p_{\text{oil}} -
\nabla_x \log p_{u}
]

This corresponds to an implicit target distribution:

[
p(x) \propto
p_{\text{dog}}(x);
p_{\text{oil}}(x)
/ p_u(x)
]

People have used this to generate “oil paintings of your dog”, even when no such images exist in any dataset.

But why does adding scores work? Why is subtracting the unconditional score necessary? And why does this work for some concepts but fail spectacularly for others?

These questions motivate the paper.

---

# **2. Why Previous Definitions of Composition Fail**

The authors argue that two common mathematical ways of defining composition are fundamentally flawed for real applications:
**the simple product** and **Bayes composition**.

Let’s unpack why.

---

## **2.1 Simple Product Composition**

The simple definition is:

[
\hat p(x) \propto p_1(x) p_2(x)
]

This seems intuitive: the composed image should be likely under *both* distributions.

The problem?

**You can never generate something that has zero probability under either distribution.**

Example:

* (p_1): images with *one* object in the lower-left
* (p_2): images with *one* object in the upper-right

Both distributions contain only single-object images.
If you take their product, any image with *two* objects has probability zero.

But composing concepts should allow out-of-distribution cases.
A combined image with *two* objects is necessary for composition.

Thus, the simple product is too restrictive.

---

## **2.2 Bayes Composition**

Prior works (Du et al. 2023, Liu et al. 2022) introduced a more principled composition:

[
p(x \mid c_1, c_2)
]

If (c_1) is “object located at lower-left”, and
(c_2) is “object located at upper-right”, then:

[
p(x \mid c_1, c_2) \propto \frac{p(x \mid c_1)p(x \mid c_2)}{p(x)}
]

This motivates the familiar score combination:

[
\nabla \log p(x \mid c_1) + \nabla \log p(x \mid c_2) - \nabla \log p(x)
]

But there is a hidden assumption:
**that the unconditional distribution contains images with both attributes at once**.

In many synthetic settings (like CLEVR), the unconditional model has never seen images with two objects.
So the formula breaks: the computed conditional doesn’t correspond to a meaningful distribution.

This leads to experiments where:

* combining location-conditioned models yields **fewer** objects than expected
* sometimes **zero** objects
* or unstable artifacts

These failures illustrate that even Bayes composition cannot justify what the model is empirically doing.

So we need a new definition of what “correct composition” even means.

---

# **3. The Paper’s Core Idea: Projective Composition**

The authors propose a fundamentally different way to define composition.

Instead of asking:

> What distribution gives high probability to all desired attributes **simultaneously**?

they ask:

> What aspects of each distribution do we want to preserve?

This leads to the idea of **projection functions**.

---

# **3.1 Projection-Based Definition**

Each distribution (p_i) comes with a projection (\Pi_i(x)).
This projection extracts the relevant feature from the sample.

For example:

* style extraction
* content extraction
* object mask extraction
* spatial region extraction (e.g., pixels around location i)
* latent feature selection

The composed distribution (\hat p) must satisfy:

[
\Pi_i \sharp \hat p = \Pi_i \sharp p_i
]

which means:

> After projecting the composed image, it should look statistically identical to the projection of images from (p_i).

This definition does **not** require:

* distributions to overlap
* support to match
* the composition to be a conditional distribution
* the concepts to be independent
* any joint distribution to exist

It only enforces that the composed result matches each source distribution *in the dimensions that matter*.

This is why the approach can create **genuinely out-of-distribution** images.

---

# **4. When Does Score Combination Actually Work? The Factorized Conditional Structure**

The next big contribution is a clean, precise condition under which:

* linear score combination
* reverse diffusion sampling

will produce a correct projective composition.

The condition is called **Factorized Conditionals**.

---

## **4.1 The Factorized Conditional Assumption**

Suppose the sample (x) lives in (n)-dimensional space.
The coordinates can be partitioned into disjoint sets:

[
M_b,; M_1,; M_2,; \ldots,; M_k
]

Each distribution (p_i):

* behaves independently across the partitioned coordinates
* modifies only the region (M_i)
* and matches a background distribution (p_b) on coordinates outside its region

Formally:

[
p_i(x) = p_i(x \mid M_i); p_b(x \mid M_i^c)
]

This means:

* The background distribution describes everything other than region (M_i).
* Model (p_i) is simply “background + special modification on coordinates in (M_i)”.

This matches the intuition in CLEVR:

* you sample an empty background,
* then “paste” an object into a small region,
* and the rest of the image stays exactly like the background.

---

## **4.2 Main Theorem: When Score Composition Works**

The paper proves:

If the distributions satisfy the factorized conditional structure,
then running reverse diffusion with the composed score:

[
\nabla_x \log p_b^t(x) +
\sum_i
\left[
\nabla_x \log p_i^t(x) - \nabla_x \log p_b^t(x)
\right]
]

will **exactly** sample from the projective composition:

[
\hat p(x) =
p_b(x \mid M_b)
\prod_i
p_i(x \mid M_i)
]

This is remarkable because:

1. It guarantees correctness under clean assumptions.
2. It shows that composition requires a *good choice of background distribution*.
3. It explains why using the unconditional distribution sometimes fails.
4. It explains why composition works best when concepts are spatially disjoint or non-interacting.

This is the first formal guarantee for score-based composition in diffusion models.

---

# **5. The Real Magic: Composition Can Happen in Feature Space**

The previous results relied on spatial partitions.
But attributes like “oil painting style” or “dog content” are not localized to pixel regions.

To generalize, the authors consider a smooth invertible mapping:

[
A : \mathbb{R}^n \to \mathbb{R}^n
]

a feature transform.

If there exists **any** feature space where the partitioned-coordinate assumption holds,
then **the same score combination in pixel space** produces a correct projective composition in that feature space.

This is the paper’s most surprising and powerful result.

### **Key Consequence:**

You don’t need to know what the correct feature space is.
Composition works automatically.

This connects composition to classical ideas like:

* disentangled representations
* coordinate-aligned latent spaces
* linear independent subspaces for different concepts

In practice, this explains why:

* style and content often compose well
* certain colors combine cleanly
* unrelated concepts can be merged if they activate orthogonal directions in feature space

---

# **6. Why Sampling Sometimes Fails Even When Composition Is Possible**

There is an important warning in the paper.

Even if a valid composed distribution exists,
and even if score addition defines it correctly,
the **reverse diffusion sampler may not be able to reach it**.

Why?

Because:

* The composition operator is not smooth.
* A small change in the constituent distributions can produce a large change in the composed distribution.
* Noise schedules interact poorly with this instability.
* The noisy distributions do not commute with the composition operator.

This captures phenomena observed in practice:

* score composition suddenly collapses for certain color pairs
* combining too many objects leads to degeneracy
* using unconditional backgrounds often produces empty images
* adding more complex attributes makes sampling unstable

The theory predicts these failures.

---

# **7. The Experiments: What Actually Works and What Fails**

The authors evaluate three major scenarios:

1. **Single objects on empty backgrounds**
2. **Single objects with unconditional backgrounds (Bayes composition)**
3. **Cluttered scenes with partial independence**

Only scenarios 1 and 3 exhibit clean composition.
Scenario 2 fails consistently.

### **7.1 Single Object Composition**

Each object occupies a different region.
The background is uniform and empty.

Factorized conditionals approximately hold.
Composition succeeds up to several objects.
Beyond 6–9 objects, failures appear due to mild violations of independence.

### **7.2 Unconditional Backgrounds**

This is the commonly used setup in previous works.

It fails because:

* the unconditional model is not the correct background distribution
* it contains objects not relevant to the desired composition
* support mismatch causes degeneracy

The result is missing objects or unstable layouts.

### **7.3 Cluttered Object Distribution**

Each conditional distribution includes:

* one specific object
* several randomly placed additional objects (clutter)

This makes the unconditional distribution resemble a mixture of all object types.
Independence becomes approximately valid.

This explains why prior text-to-image composition sometimes works—even though the theory seems inapplicable.

---

# **8. Predicting Whether a Composition Will Work: A Practical Heuristic**

Although the theory is abstract, the authors give a beautifully simple heuristic.

If concept vectors are approximately orthogonal, then composition is likely to work.

To test this:

1. Take a set of images representing concept (i).
2. Compute their CLIP embeddings.
3. Subtract the background embedding.
4. Compare the vectors for different concepts.

If

[
(\mu_i - \mu_b)^T (\mu_j - \mu_b) \approx 0
]

then concepts (i) and (j) should compose well.

This aligns perfectly with empirical observations:

* Style + content compose cleanly.
* Dog + hat compose cleanly.
* Dog + horse does not.
* Colors in different channels can combine.
* Entangled attributes (two subjects) fail.

This is one of the most actionable takeaways of the paper.

---

# **9. Broader Implications**

This work provides a rigorous basis for understanding and designing new compositional generative systems.

### **9.1 Understanding OOD Generalization**

The theory formalizes when diffusion models can go out-of-distribution reliably.
This helps explain phenomena like:

* length generalization
* layout extrapolation
* stylistic transformations

### **9.2 Towards Modular Generative Models**

By clarifying how independent generative components can be glued together,
the work supports larger goals like:

* plug-and-play generative modules
* compositional world models
* multi-agent generative reasoning
* modular scene construction

### **9.3 Future Directions**

The biggest open problem remains:

**sampling in general feature spaces is still difficult.**

We need:

* better samplers
* noise-aware composition operators
* ways to learn feature spaces with factorized structure
* improved background estimation strategies
* theoretical insights into composition stability

This work lays a foundation.
But composition is still an emerging frontier.

---

# **10. Conclusion**

The paper *Mechanisms of Projective Composition of Diffusion Models* provides a powerful, elegant answer to a fundamental mystery:
How do diffusion models combine different concepts into a coherent image?

The key insights are:

1. Prior definitions of composition are flawed because they cannot generate out-of-distribution samples.
2. A new definition, **projective composition**, captures how concept-level attributes should combine.
3. Under **factorized conditional** structure, score addition and reverse diffusion yield exact compositions.
4. Correct feature spaces exist where composition is clean—even if we never explicitly find them.
5. Sampling may fail when noise interacts poorly with the composition operator.
6. A simple CLIP-based orthogonality test predicts success or failure.

Together, these contributions turn previously empirical observations into a coherent mathematical theory.

Diffusion models will continue to move toward modularity, controllability, and compositional expressiveness.
This paper takes a large step toward understanding the principles that make this possible.

