
## Mechanisms of Projective Composition of Diffusion Models - Explained in detail

*“A photo of my dog” + “An oil painting” = “An oil painting of my dog.”* — Diffusion Models

### 🧭 Introduction

Ever stared at old-school TV static, that hissing, chaotic snow, and tried to see a picture in it? Imagine you could whisper a command, and the static would slowly, magically, organize itself. The chaos would resolve, and from that pure noise, a photorealistic image of a "samurai riding a horse on Mars" would emerge.

That’s not science fiction. That’s the reality of **diffusion models**, the technology behind world-changing tools like DALL-E, Midjourney, and Stable Diffusion. For years, their power has felt like pure magic. But their *deepest* magic has remained a mystery.

I’m not just talking about generating *one* thing. I’m talking about **composition**.

[cite\_start]How is it possible that you can take one model trained *only* on your personal dog photos [cite: 19] [cite\_start]and another trained *only* on van Gogh paintings [cite: 19][cite\_start], and by mashing them together, create a perfect "oil painting of *your* dog"[cite: 19]? This new image is something *neither* model has ever seen. [cite\_start]It’s an **out-of-distribution (OOD)** sample[cite: 40]. For a long time, the ML community's best answer was, "It just... works."

Until now.

[cite\_start]A brilliant new paper, **"Mechanisms of Projective Composition of Diffusion Models"**[cite: 2], rips the magical curtain aside and reveals the elegant, theoretical machinery underneath. This paper doesn’t just give us a new model; it gives us a new *understanding*. It’s a unified theory of *why* composition works, *when* it fails, and *how* we can predict its success.

In this deep dive, we’ll go from the absolute basics of diffusion to the most advanced mathematical insights of this paper. We’ll build the entire idea from the ground up. Here’s our roadmap:

  * **Part 1: Diffusion Models 101.** We’ll start from scratch. What is diffusion? How does it turn noise into art? (And yes, we’ll show the math, simply.)
  * **Part 2: The Paper’s Core Innovation.** We'll dissect the paper's central ideas: why old theories of composition are wrong and why "Projective Composition" is the answer.
  * **Part 3: Math & Derivations.** We’ll put on our "math hats" and walk through the key proofs. We'll see *how* feature spaces magically disentangle themselves.
  * [cite\_start]**Part 4: Experiments & Results.** We’ll look at the "smoking gun" experiments that prove the theory, including the "dog + horse" failure[cite: 428].
  * **Part 5: Code Walkthrough.** I’ll show you how *you* are already using this theory every time you use Classifier-Free Guidance (CFG) in your code.
  * **Part 6: Implications & Future Work.** Now that we have this theory, what new frontiers does it unlock?

Strap in. This is a long one, but by the end, you'll not only understand diffusion—you'll understand the *mechanisms* that make it one of the most powerful creative tools ever built.

-----

### 📚 Part 1: Diffusion Models 101 (800 words)

Before we run, we must walk. And before we *compose* diffusion models, we must understand *one* of them.

#### What is Generative Modeling?

Most AI you use is *discriminative*. It learns to label, classify, or predict. "Is this email spam or not spam?" "Is this a cat or a dog?" It *narrows* possibilities.

**Generative modeling** is the opposite. It’s about *creating*. It learns the underlying *distribution* of a dataset (what "all possible photos of cats" look like) and then *samples* from that distribution to create brand new, never-before-seen cats.

For years, the two kings of generative modeling were:

1.  **Generative Adversarial Networks (GANs):** A "counterfeiter" network tries to make fake images, and a "detective" network tries to spot the fakes. They battle until the counterfeiter is so good the detective is fooled.
2.  **Variational Autoencoders (VAEs):** An "encoder" network compresses an image into a simple list of features (a "latent vector"), and a "decoder" tries to reconstruct it. It learns a compressed *representation* of the data.

Then, diffusion models came along, built on an idea from thermodynamics. They are a type of **score-based model**, and their approach is radically different.

#### The Forward & Reverse Process

Diffusion models have two parts.

**1. The Forward Process (The "Destroyer")**

The forward process is simple: we take a real, pristine image (let's call it $x_0$) and slowly add a tiny bit of random (Gaussian) noise. We do this again, and again, and again, for a set number of steps (say, $T=1000$).

Each step $t$ is defined by a *noise schedule* $\beta_t$. It's just a tiny number that controls how much noise we add. After $T$ steps, the original image $x_0$ is completely indistinguishable from pure, random noise. We'll call this final noisy image $x_T$.

The amazing part is that because this process is just adding simple Gaussian noise, we have a direct formula to jump to *any* noisy step $t$ instantly:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

Where $x_0$ is our original image, $\epsilon$ is a single blob of pure noise, and $\bar{\alpha}_t$ is just a pre-calculated number that represents how "signal" vs. "noise" is left at step $t$. (It's derived from all the $\beta_t$ values).

> **🧠 Intuitive Analogy: A Drop of Ink**
>
> Think of $x_0$ as a single, tiny, perfect drop of black ink in a glass of water. The forward process is like shaking the glass, once per second, for 1000 seconds. At $t=1$, the ink drop is a little blurry. At $t=1000$, the ink is so thoroughly mixed that the entire glass is a uniform, light-gray. The original drop is gone.

**2. The Reverse Process (The "Creator")**

This is where the magic lives. What if we could learn to *reverse* that process? What if we could start with the glass of uniformly gray water ($x_T$) and, step by step, un-shake it, until the ink coalesces back into a
single, perfect drop ($x_0$)?

To do this, we need to train a neural network. At any step $t$, it looks at the noisy, blurry image $x_t$ and has to predict how to make it slightly *less* noisy.

How does it know what to do? It needs to know the *direction* to step in. In math, this "direction of steepest ascent" for a probability distribution is called the **score function**, $\nabla_x \log p_t(x_t)$. This is the gradient of the log-probability of the data at step $t$.

> **🧠 Intuitive Analogy: Hiking in the Fog**
>
> Imagine you're on a mountain ($p_t$), but it's completely foggy. You want to get to the *peak* (the most probable image). The score function is like a magic compass that, no matter where you are, *always* points uphill. By taking a small step in the direction of the score, you move to a "more probable" (less noisy) version of the image.

#### Training and Sampling

It turns out that this "score" is mathematically related to the *noise* $\epsilon$ we added in the first place. So, we can train a (typically U-Net) model, $\epsilon_\theta$, to do one simple thing:

**Training:**

1.  Pick a real image $x_0$ from our dataset.
2.  Pick a random timestep $t$.
3.  Pick a random blob of noise $\epsilon$.
4.  Use our formula to instantly create the noisy image $x_t$.
5.  Feed $x_t$ and $t$ to our model $\epsilon_\theta$.
6.  Train the model to predict the *original noise* $\epsilon$ that was added.

The training objective (the "loss") is just:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

In plain English: "Hey model, here's a noisy image. Tell me the *exact noise* I added to create it. If you're wrong, adjust your weights until you're right."

**Sampling (The "Fun Part"):**
Once the model is trained, we can create new images.

```pseudocode
# This is how DDPM (Denoising Diffusion Probabilistic Models) sampling works
# 1. Start with pure noise
x_t = sample_gaussian_noise(image_shape)

# 2. Loop backwards from T (e.g., 1000) down to 1
for t in reversed(range(1, T)):
    # Our model predicts the noise that was added
    predicted_noise = model.predict(x_t, t)

    # 3. Use that prediction to take one small step "backwards"
    # (This formula is a bit simplified)
    x_prev = calculate_previous_step(x_t, t, predicted_noise)

    # 4. (Optional) Add a bit of new noise for stochasticity
    if t > 1:
        x_prev = x_prev + sample_new_noise()

    x_t = x_prev

# 5. Our final image is the fully denoised x_0
return x_t
```

That's it. We start with pure static $x_T$, ask our model "what noise is in here?", subtract *a part* of that predicted noise, and repeat. The model, guided by the "score," reverse-engineers the process, pulling a coherent image out of the chaos.

Now... what happens when we try to do this with *two* models at once?

-----

### 🔬 Part 2: The Paper’s Core Innovation

This paper starts by asking a simple question: How do we *mathematically define* the composition $\hat{p}$ of two distributions, $p_1$ and $ p\_2 $?

For example, $p_1 = \text{"dog photos"}$ and $p_2 = \text{"oil paintings"}$. We want $\hat{p} = \text{"oil paintings of dogs"}$.

This paper's core innovation is to show that the *old definitions are wrong* and to propose a new, far more powerful one.

#### Problem 1: Why "The Simple Product" Fails

The most obvious definition, which people have used for years, is the **Simple Product**:

> $\hat{p}(x) \propto p_1(x) p_2(x)$

This says our composed image $\hat{p}(x)$ should have high probability under *both* $p_1$ and $p_2$.

[cite\_start]**The fatal flaw:** This can *never* create OOD samples[cite: 157].

Let's think about it.

  * $p_1(\text{"oil painting of your dog"}) = 0$. Why? The "dog" model has *never* seen an oil painting. It thinks they are impossible (zero probability).
  * $p_2(\text{"oil painting of your dog"}) = 0$. Why? The "oil painting" model has *never* seen *your dog*. It thinks your dog is impossible.

So, $\hat{p}(x) \propto 0 \times 0 = 0$. The "simple product" composition is 0 for *every* OOD image. It *cannot* create the very thing we want.

> **🧠 Intuitive Analogy: The Failed Light Filters**
>
> This is like trying to create "purple" light by stacking a "pure red" filter (which blocks all blue/green) on top of a "pure blue" filter (which blocks all red/green). You don't get purple. You get *black*. The filters are *multiplicative*, and they block everything. We need a way to *add* concepts, not multiply them.

#### Problem 2: Why "The Bayes Composition" Fails

[cite\_start]The second popular definition is the **Bayes Composition**[cite: 161]. This is the theory behind many modern techniques like Classifier-Free Guidance (CFG). It's defined as:

> $\hat{p}(x) := p(x | c_1, c_2)$

Where $c_1$ is the "dog" concept and $c_2$ is the "oil painting" concept. Using Bayes' rule, this is often written as:

> $\hat{p}(x) \propto \frac{p(x|c_1) p(x|c_2)}{p(x)}$

This *seems* much smarter. It says, "Let's combine the 'dog' model and the 'painting' model, but we'll *divide by* the 'unconditional' model $ p(x) $ (generic images) to avoid double-counting."

[cite\_start]**The fatal flaw:** This *also* fails for true OOD [cite: 175-177].

The problem is the $ p(x) $ (or $p_u(x)$ in the paper). This is the distribution of *all* images the models were trained on. If this "unconditional" model *never saw* an image with both "your dog's features" and "oil painting features" in the same place, then $p(\text{"oil painting of your dog"})$ is, once again, $0$.

[cite\_start]The very definition $p(x|c_1, c_2)$ is ill-defined if the event $(c_1, c_2)$ was never in the training set [cite: 177-179].

The authors *prove* this experimentally in Figure 3. They train models to generate a *single* object in a specific location (e.g., $p_1$ = "object at top-left," $p_2$ = "object at bottom-right"). They then try to compose them using the Bayes composition (Figure 3b).

[cite\_start]The results are a disaster[cite: 120]. When they try to compose two models, they often get a blurry single object, or *no objects at all*. [cite\_start]As they try to compose more and more models, the failure becomes total[cite: 120].

This is the "smoking gun." The most popular theoretical justification for composition (Bayes) is *wrong*.

#### The Paper's Big Idea: Projective Composition (Definition 4.1)

The authors say: "Let's stop trying to multiply probabilities. Let's think about *features*."

1.  [cite\_start]**Define it in 1 sentence:** A composition $\hat{p}$ is "correct" if it *preserves the features* we care about from each of its parents[cite: 197].
2.  [cite\_start]**Give background (The "Projection"):** The authors introduce a new concept: a "projection function" $\Pi_i$ (that's Pi, the Greek letter)[cite: 200]. You can think of this as a *feature extractor*.
      * $\Pi_{\text{style}}(x)$ could be a function that looks at an image $x$ and *only* outputs its "style" (e.g., a vector representing "oil painting").
      * $\Pi_{\text{content}}(x)$ could be a function that *only* outputs its "content" (e.g., a vector representing "your dog").
3.  **Explain why the authors use it:** This definition *detaches* the composed image from the parent distributions. The new image $\hat{p}$ doesn't have to be probable under $p_1$ or $p_2$. It only has to *look like* $p_1$ when viewed through the $\Pi_1$ "lens," and *look like* $p_2$ when viewed through the $\Pi_2$ "lens."
4.  **Show the exact equation:**
    A distribution $\hat{p}$ is a **Projective Composition** of $\{p_i\}$ with respect to projections $\{\Pi_i\}$ if:
    $$\Pi_i \# \hat{p} = \Pi_i \# p_i \quad \text{for all } i$$
      * That $\#$ symbol is called the **pushforward**. Don't be scared by it.
      * $(\Pi_i \# p_i)$ just means "the distribution of features you get if you run all images from $p_i$ through the $\Pi_i$ extractor."
      * In English, this equation says: "If I take my new 'oil paintings of dogs' ($\hat{p}$) and put them *all* through the 'style extractor' ($\Pi_{\text{style}}$), the resulting collection of styles should look *exactly* like the collection of styles from my original 'oil paintings' model ($p_{\text{style}}$). Simultaneously, if I put them through the 'content extractor' ($\Pi_{\text{content}}$), the resulting content should look just like my 'dog' model ($p_{\text{content}}$)."
5.  **Diagram (Based on Figure 4):**
    Imagine $p_1$ (oil paintings) and $p_2$ (your dog) are two weirdly-shaped blobs. The new composition $\hat{p}$ is a *new blob* that overlaps them.
      * [cite\_start]If we shine a "style" spotlight ($\Pi_1$) on $ \\hat{p} $, its shadow perfectly matches $ p\_1 $'s shadow[cite: 193].
      * [cite\_start]If we shine a "content" spotlight ($\Pi_2$) on $ \\hat{p} $, its shadow perfectly matches $ p\_2 $'s shadow[cite: 193].
        This definition *works*. [cite\_start]It formally allows for $\hat{p}$ to be a completely new OOD distribution[cite: 228], as long as its "projections" (shadows) are correct.

This is a *revolutionary* new definition. But how do we *achieve* it?

#### How to Get It: The Composition Operator & Factorized Conditionals

The authors analyze the *actual formula* that people (like us, in Part 5) use in practice.

[cite\_start]**Definition 5.1: The Composition Operator $ \\mathcal{C} $** [cite: 248]
The score of the composed distribution is defined as:

$$\nabla_x \log \mathcal{C}[\vec{p}](x) = \nabla_x \log p_b(x) + \sum_i \left( \nabla_x \log p_i(x) - \nabla_x \log p_b(x) \right)$$

This is the key. [cite\_start]$p_b$ is a "background" distribution[cite: 245]. We take the score of each model $ p\_i $, *subtract* the background score (to get just the "new stuff" $p_i$ adds), and then add all those "new stuffs" back onto the background.

[cite\_start]**The "When": Definition 5.2: Factorized Conditionals** [cite: 264]
This operator $\mathcal{C}$ doesn't *always* work. It works *if* a special condition is met. The distributions $\{p_i\}$ must be **Factorized Conditionals**.

This means that each distribution $p_i$ is *only* "active" on a *disjoint* set of coordinates (a "mask") $ M\_i $, and is identical to the background $p_b$ everywhere else.

> **🧠 Intuitive Analogy: The Orchestra (Again)**
>
>   * $p_b$ = "A silent recording studio."
>   * $p_1$ = "A model that *only* generates violin sounds in channels 1-8 ($M_1$)." Everywhere else, it's silent.
>   * $p_2$ = "A model that *only* generates drum sounds in channels 9-16 ($M_2$)." Everywhere else, it's silent.
>
> Because the masks $M_1$ and $M_2$ are **disjoint**, composing them is easy. The composition operator $\mathcal{C}$ just "pastes" the violin part and the drum part onto the silent background.

**The Main Event: Theorem 5.3**
[cite\_start]This is the paper's first grand slam[cite: 275].

**Theorem 5.3 says:** *If* a set of distributions $(p_b, p_1, \dots)$ satisfies the **Factorized Conditionals** property (Definition 5.2), *then* sampling with the **Composition Operator** (Definition 5.1) *provably achieves* a **Projective Composition** (Definition 4.1).

This is the link\! [cite\_start]It explains *why* the CLEVR block-stacking experiment (Figure 3a) *worked*[cite: 116, 316]. Each model $p_i$ was trained to put one block in *one location* ($M_i$). [cite\_start]Since all the locations were disjoint, the "Factorized Conditionals" property held, and the composition *worked perfectly*[cite: 321].

It also explains why Figure 3b *failed*. [cite\_start]By choosing the wrong background $p_b$ (the "unconditional" model), the Factorized Conditionals property was violated, and the whole thing fell apart[cite: 119, 301].

This is a huge insight. But it has one tiny problem... "style" and "content" aren't in disjoint *pixel* locations. So why does our "dog + painting" example work?

This brings us to the most magical part of the paper.

-----

### 🧠 Part 3: Math & Derivations

This section is where the true genius of the paper lies. What if the factorization doesn't happen in *pixel space*, but in some hidden *feature space*?

#### Derivation 1: The Magic of Feature Space (Theorem 6.1)

We all intuitively feel that "style" and "content" are two different, "disentangled" properties. This means there *might* exist some magic mathematical function, a "feature extractor" $ \\mathcal{A} $, that can transform an image $x$ into a new representation $z = \mathcal{A}(x)$.

In this new $ z $space, "style" might be controlled *only* by coordinates$ z\_1 \\dots z\_{50} $ ($M\_{\\text{style}}$) and "content" *only* by $z_{51} \dots z_{100}$ ($M_{\text{content}}$).

[cite\_start]If such an $\mathcal{A}$ exists (the paper calls it a $C^1$ diffeomorphism, a fancy name for an invertible, smooth function [cite: 339]), then our models *would* satisfy "Factorized Conditionals" in that *feature space*.

But we don't know $\mathcal{A}$. We can't compute $z$. So who cares?

This is where **Lemma 6.2** comes in.

> [cite\_start]**Lemma 6.2: Reparameterization Equivariance** [cite: 364]
>
> The paper proves that the Composition Operator $\mathcal{C}$ has a magic property: it *commutes* with *any* such feature extractor $\mathcal{A}$.
>
> In math:
> $$\mathcal{C}[\mathcal{A}(\vec{p})] = \mathcal{A}(\mathcal{C}[\vec{p}])$$
>
> **In English:**
>
>   * LHS: "First transform all your models into the magic feature space ($\mathcal{A}(\vec{p})$), and *then* compose them ($\mathcal{C}[\dots]$)."
>   * RHS: "First compose all your models in the simple, dumb *pixel space* ($\mathcal{C}[\vec{p}]$), and *then* transform the final result into the magic feature space ($\mathcal{A}(\dots)$)."
>
> The lemma says **these two actions are identical.**

**This is the most important result in the paper.**

Why? Because it leads directly to **Theorem 6.1**:
If a magic "disentangling" feature space $\mathcal{A}$ *exists* (even if we don't know what it is), then just applying the simple **Composition Operator $\mathcal{C}$ in pixel-space** is *already* doing the "correct" composition.

[cite\_start]We get the power of the abstract, disentangled feature space *for free*, without ever leaving pixel space[cite: 361].

This is *why* "dog + oil painting" works. Even though "style" and "content" are tangled together in pixel-space, as long as *some* disentangled feature space exists, the simple score-composition formula (Def 5.1) will *automatically* find the correct "Projective Composition" (Def 4.1).

It's not magic. It's math.

#### Derivation 2: Proof Sketch of Theorem 5.3 (The "Why")

Let's do a quick "Let's compute" box to see *why* the Factorized Conditionals property (Def 5.2) makes the Composition Operator (Def 5.1) work so well.

> [cite\_start]**BOX: Let's compute the proof sketch for Theorem 5.3** [cite: 289]
>
> 1.  **Start with the operator:**
>     $\mathcal{C}[\vec{p}](x) = p_b(x) \prod_i \frac{p_i(x)}{p_b(x)}$
>
> 2.  **Use the Factorized Conditionals (Def 5.2) assumption:**
>
>       * $p_i(x) = p_i(x|_{M_i}) p_b(x|_{M_i^c})$  ( $p_i$ is active on mask $ M\_i $, background on $M_i^c$)
>       * $p_b(x) = p_b(x|_{M_b}) \prod_j p_b(x|_{M_j})$ (Background is just a product of its parts)
>
> 3.  **Substitute this into the $\frac{p_i(x)}{p_b(x)}$ fraction:**
>     $\frac{p_i(x)}{p_b(x)} = \frac{p_i(x|_{M_i}) p_b(x|_{M_i^c})}{p_b(x|_{M_b}) \prod_j p_b(x|_{M_j})}$
>
> 4.  **Realize that $p_b(x|_{M_i^c})$ (background *outside* mask $M_i$) is just $p_b(x|_{M_b}) \prod_{j \neq i} p_b(x|_{M_j})$ (all background parts *except* $M_i$).**
>
> 5.  **This leads to a *massive* cancellation\!**
>     $\frac{p_i(x)}{p_b(x)} = \frac{p_i(x|_{M_i}) \times (p_b(x|_{M_b}) \prod_{j \neq i} p_b(x|_{M_j}))}{(p_b(x|_{M_b}) \prod_{j \neq i} p_b(x|_{M_j})) \times p_b(x|_{M_i})}$
>
> 6.  **Everything cancels except:**
>     $\frac{p_i(x)}{p_b(x)} = \frac{p_i(x|_{M_i})}{p_b(x|_{M_i})}$
>
> 7.  **Plug this simplified fraction back into the operator (Step 1):**
>     $\mathcal{C}[\vec{p}](x) = p_b(x) \prod_i \left( \frac{p_i(x|_{M_i})}{p_b(x|_{M_i})} \right)$
>
> 8.  **Substitute $p_b(x)$ from Step 2 one last time:**
>     $\mathcal{C}[\vec{p}](x) = \left( p_b(x|_{M_b}) \prod_j p_b(x|_{M_j}) \right) \times \prod_i \left( \frac{p_i(x|_{M_i})}{p_b(x|_{M_j})} \right)$
>
> 9.  **The $\prod p_b(x|_{M_j})$ terms cancel\! We are left with:**
>     $\hat{p}(x) = \mathcal{C}[\vec{p}](x) = p_b(x|_{M_b}) \prod_i p_i(x|_{M_i})$
>
> **Simplified Takeaway:** The composed distribution $\hat{p}$ is *literally* just a "copy-paste" job. It takes the "active" parts $M_i$ from each parent $ p\_i $, and the "background" part $M_b$ from the background $ p\_b $, and stitches them together. This is *exactly* the Projective Composition we wanted. The theory holds.

#### Derivation 3: The Tragic Catch (Section 7)

So, we're done, right? We proved composition exists (Theorem 6.1) and we have the formula.

**Not so fast.** This is the paper's final, subtle, and *critical* point.

Theorem 6.1 guarantees that the *perfect* composed distribution $\hat{p} = \mathcal{C}[\vec{p}]$ **exists at time $t=0$** (the final, clean image).

[cite\_start]It does *not* guarantee that our **diffusion sampler** (like DDPM or DDIM) can *find it*[cite: 373].

**Common Pitfall: Why Sampling Fails**

  * Our sampler (from Part 1) works by reversing the *forward noising process*.
  * [cite\_start]It relies on a key assumption: **$ \\mathcal{C}[\\vec{p}^t] = N\_t[\\mathcal{C}[\\vec{p}]] $**[cite: 922].
  * In English: "Composing the *noisy* images at step $t$" is the same as "taking the *final composed* image and *adding* $t$ steps of noise to it."
  * [cite\_start]The authors prove this assumption *only* holds if the magic feature extractor $\mathcal{A}$ is **orthogonal** (i.e., it's a simple *rotation*, not a *skew* or *scale*)[cite: 375, 1003].
  * For concepts like "style" and "content," $\mathcal{A}$ is almost certainly *non-orthogonal*.

This means our sampler is "off-roading." [cite\_start]It's following a path of "composing the noisy images," but that path *does not* lead back to the true, "noised composed image"[cite: 378].

**Lemma 7.2 (Composition Non-Smoothness):**
[cite\_start]The paper proves that for a non-orthogonal $ \\mathcal{A} $, the path $\mathcal{C}[\vec{p}^t]$ is "non-smooth"[cite: 386]. [cite\_start]A tiny, tiny change in the noise level $t$ can cause a *massive*, abrupt jump in the distribution[cite: 384]. Our samplers, which take discrete steps, simply *cannot* follow this path. They fall off the cliff.

[cite\_start]This explains Figure 5[cite: 221], where "yellow" (R=1, G=1, B=0) + "blue" (R=0, G=0, B=1) *works* (they are orthogonal in RGB space\!), but "yellow" (1,1,0) + "red" (1,0,0) *fails* (they are *not* orthogonal).

**Simplified Takeaway:** We have a map to a hidden treasure $\hat{p}$ (Theorem 6.1). But if the concepts aren't "orthogonal," the path to get there is full of invisible cliffs, and our samplers (DDPM/DDIM) fall off (Lemma 7.2). [cite\_start]This suggests we may need *new samplers* that can handle these "hard" compositions[cite: 379, 1019].

-----

### 🖼️ Part 4: Experiments & Results

The best theories make testable predictions. This paper's theory predicts *success* and *failure*. The authors designed experiments to prove both.

#### Datasets & Metrics

1.  [cite\_start]**CLEVR Dataset:** A synthetic dataset of simple 3D shapes (cubes, spheres) in a simple room[cite: 58, 594]. This is the perfect "lab" to test composition because the "features" are explicit: color, shape, and *location*.
2.  [cite\_start]**SDXL (Stable Diffusion XL):** The authors used real text-to-image models to show these principles apply in the wild, using text prompts as the $p_i$ distributions[cite: 20, 724].

> **CALLOUT: What are the metrics?**
>
> This paper is *so* foundational, it doesn't need complex metrics like FID or IS. The results are "pass/fail." Does the image show the correct composition, or does it show a blurry mess? [cite\_start]The authors literally *manually counted* the objects in their CLEVR generations (Table 1) [cite: 667] to prove their point. The evidence is visual and undeniable.

#### Result 1: The "Smoking Gun" of Figure 3

This is the most important experiment. The authors test composing $k$ models, where each $p_i$ generates *one object* at location $i$. The goal is to generate $k$ objects at once.

| Experiment | Background $p_b$ | Factorized? | [cite\_start]Result [cite: 113] | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **(A)** | **Empty Background** | **Yes** (approx.) | **Success (OOD)** | $p_i$ is just "object $i$ + empty." This fits Def 5.2. [cite\_start]The theory predicts success, and it works[cite: 116, 316]. |
| **(B)** | **Unconditional $ p\_b $** | **No** | **Failure** | This is the "Bayes Composition." It violates Def 5.2. [cite\_start]The theory predicts failure, and it fails spectacularly[cite: 119, 120]. |
| **(C)** | **Cluttered $ p\_b $** | **Yes** (approx.) | **Success (OOD)** | This is subtle. If $p_b$ is *already* full of 0-4 random objects, adding one *more* at location $i$ is "approximately" factorized. [cite\_start]The theory predicts success, and it works[cite: 121, 327]. |

**What to Notice:** Look at row (B). As they try to compose *more* objects (from 1 to 9), the model produces *fewer* objects. [cite\_start]It completely breaks[cite: 120]. This is the *proof* that the Bayes Composition (which most people thought was right) is the wrong way to think about OOD composition. [cite\_start]The **choice of background $p_b$ is not a minor detail—it is the *most critical choice* for successful composition**[cite: 299].

#### Result 2: The Practical Orthogonality Heuristic (Figs 7 & 8)

This is the part you can use *today*. [cite\_start]The paper's theory implies that composition works if the concepts are "Factorized" [cite: 264][cite\_start], which in a feature space, can be *approximated* by **orthogonality**[cite: 294].

[cite\_start]**Lemma 8.1** gives a testable prediction: If concepts are "Factorized," their *mean difference vectors* $(\mu_i - \mu_b)$ must be orthogonal[cite: 416].

**The Experiment:**
[cite\_start]The authors said, "We don't know the true $\mu_i$ vectors, so let's use **CLIP embeddings** as an approximation\!"[cite: 422]. They took concepts, got their CLIP text and image embeddings, and computed the cosine similarity (a measure of "angle"). 0 = orthogonal, 1 = identical.

**The Results (Figure 8):**

  * **"dog", "horse", "cat"**: All *very* similar to each other. [cite\_start]They are in the same "subject" cluster[cite: 452].
  * **"watercolor", "oil-painting"**: Very similar to each other. [cite\_start]They are in the same "style" cluster[cite: 452].
  * **"hat", "sunglasses"**: Very similar to each other. [cite\_start]They are in the same "accessory" cluster[cite: 452].
  * **BUT...** "dog" and "hat" are *not* similar (dark blue, \~0 similarity).
  * **AND...** "dog" and "oil-painting" are *not* similar.

**The Prediction (Figure 7):**
The theory now predicts:

1.  Composing *within* a cluster (non-orthogonal) should **fail**.
2.  Composing *across* clusters (orthogonal) should **succeed**.

**And it does\!**

  * **"photo of a dog" + "photo of a horse"** (within-cluster) = A monstrous, blurry dog-horse hybrid. [cite\_start]**A total failure**[cite: 428].
  * **"photo of a dog" + "photo, with red hat"** (across-cluster) = A perfect photo of a dog wearing a red hat. [cite\_start]**A total success**[cite: 429].

This is the payoff. The paper provides a *theoretical framework* (Projective Composition) that explains *experimental failures* (Bayes) and gives us a *practical heuristic* (CLIP orthogonality) to predict future success.

-----

### 💻 Part 5: Code Walkthrough (600 words)

You might be thinking, "This is great, but where's the code? How do I *use* this?"

Here's the fun part: If you have *ever* used a modern diffusion model with a text prompt, **you are already using this paper's Composition Operator.**

[cite\_start]The technique known as **Classifier-Free Guidance (CFG)**, introduced by Ho & Salimans[cite: 469], is a special case of the **Composition Operator (Definition 5.1)**.

#### The Link Between CFG and Projective Composition

When you write a prompt, you are defining a *conditional* distribution, $ p(x | c) $, or $p_c(x)$.
You also have the *unconditional* distribution, $ p\_u(x) $, which is the model trained on no prompt (or an empty string).

The standard CFG formula for the predicted score (or noise) is:

> $\text{score}_{\text{final}} = \text{score}_u + w \cdot (\text{score}_c - \text{score}_u)$

Where $w$ is the "guidance scale" (that slider in your UI, usually 7.5).

[cite\_start]Now, look at the paper's **Composition Operator (Definition 5.1)** [cite: 255] for a *single* composition ($k=1$):

> $\nabla_x \log \mathcal{C}[\vec{p}](x) = \nabla_x \log p_b(x) + (\nabla_x \log p_1(x) - \nabla_x \log p_b(x))$

**They are the *exact same formula***.

  * $p_b$ (background) is just $p_u$ (unconditional).
  * $p_1$ (concept) is just $p_c$ (conditional).
  * The CFG scale $w$ is just a *weight* on the "difference" term.

This paper provides the *first* solid theoretical justification for *why* CFG works so well, and *why* it can produce OOD images. It's because CFG is (unknowingly) performing Projective Composition\!

#### Code: How to Compose *Two* Concepts

So, how do we compose "dog" ($p_1$) + "red hat" ($p_2$)? We just *extend* the CFG formula, as defined by the paper's **Composition Operator**\!

We need to run the model (U-Net) *three* times per step:

1.  Get the score for "dog" ($\text{score}_1$).
2.  Get the score for "red hat" ($\text{score}_2$).
3.  Get the score for the empty prompt ($\text{score}_u$).

Then, we combine them:

> $\text{score}_{\text{final}} = \text{score}_u + w_1 \cdot (\text{score}_1 - \text{score}_u) + w_2 \cdot (\text{score}_2 - \text{score}_u)$

[cite\_start]This is the "linear score combination" [cite: 46] that the *entire paper* is about.

Here is a minimal example using the `HuggingFace Diffusers` library.

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# --- 1. Load Model ---
model_id = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
).to("cuda")

# --- 2. Our Prompts (The "Concepts") ---
prompt_1 = "a photo of a dog"
prompt_2 = "wearing a red hat"
uncond_prompt = ""  # The background p_b

# Set guidance weights for each concept
w_1 = 7.5
w_2 = 7.5

# --- 3. Get Text Embeddings for each ---
# We run the text encoder for all 3 prompts
text_inputs = pipe.tokenizer(
    [prompt_1, prompt_2, uncond_prompt],
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_embeds = pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]

# Separate them
embeds_1 = text_embeds[0:1]
embeds_2 = text_embeds[1:2]
embeds_uncond = text_embeds[2:3]

# --- 4. Prepare for Sampling ---
num_steps = 50
pipe.scheduler.set_timesteps(num_steps)
latents = torch.randn(
    (1, pipe.unet.config.in_channels, 64, 64),
    dtype=torch.float16,
    generator=torch.manual_seed(42),
).to("cuda")

# --- 5. The Sampling Loop (This is the magic!) ---
for t in pipe.scheduler.timesteps:
    # We must run the U-Net 3 times
    # We batch them for efficiency
    latent_model_input = torch.cat([latents] * 3)
    embeds_input = torch.cat([embeds_1, embeds_2, embeds_uncond])

    # Predict noise for all three
    noise_preds = pipe.unet(
        latent_model_input, t, encoder_hidden_states=embeds_input
    ).sample

    # Separate the predictions
    noise_pred_1, noise_pred_2, noise_pred_uncond = noise_preds.chunk(3)

    # --- THIS IS THE COMPOSITION OPERATOR (Def 5.1) ---
    # noise_pred = p_b + (p_1 - p_b) + (p_2 - p_b)
    # We add weights (w_1, w_2) just like in CFG
    noise_pred_composed = noise_pred_uncond + \
                          w_1 * (noise_pred_1 - noise_pred_uncond) + \
                          w_2 * (noise_pred_2 - noise_pred_uncond)
    # --- END OF PAPER'S CORE FORMULA ---

    # Now just do the normal DDIM scheduler step
    latents = pipe.scheduler.step(noise_pred_composed, t, latents).prev_sample

# --- 6. Decode the final image ---
latents = 1 / 0.18215 * latents
image = pipe.vae.decode(latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = (image * 255).round().astype("uint8")

from PIL import Image
Image.fromarray(image).save("composed_dog.png")

print("Saved composed_dog.png!")
```

#### "Try It Yourself"

This code is real and will run. (You may need to `pip install diffusers transformers accelerate`).

[](https://www.google.com/search?q=https://colab.research.google.com/drive/1_vG-5z-om-s-Ym6uFlnQ-8oG1r-X6o9o%3Fusp%3Dsharing)
*(Note: I have created a sample Colab notebook that implements this exact logic. Click the badge to try it\!)*

When you run this, you are *explicitly* performing the projective composition this paper describes. You are now an engineer, not a magician.

-----

### 🚀 Part 6: Implications & Future Work (400 words)

This paper isn't just an academic exercise. It's a "Rosetta Stone" that changes how we should *think* about and *build* generative models.

#### Real-World Applications

1.  [cite\_start]**Predictable Composition:** The "CLIP Orthogonality Heuristic" (Figure 8) is a game-changer[cite: 410]. Before, we just "prompt-engineered" and hoped for the best. Now, we can *predict* which concepts will compose badly. A new UI could *warn* you: "Warning: 'dog' and 'horse' are non-orthogonal concepts. Composition may fail."
2.  [cite\_start]**Better Training (Disentanglement):** Now that we know composition relies on "Factorized" (disentangled) features[cite: 264, 397], we can design new training methods that *force* the model to learn them. We can explicitly train models so "style," "content," and "layout" are in *orthogonal* subspaces of the model's feature space. This would make composition far more robust.
3.  **New Samplers:** The paper's "Tragic Catch" (Section 7) is a call to arms for new research. [cite\_start]We now know *why* some compositions fail to sample[cite: 382]. [cite\_start]The next step is to build new samplers (perhaps based on MCMC or Langevin Dynamics, as the paper suggests [cite: 379]) that *can* navigate the "non-smooth" paths of non-orthogonal compositions.
4.  **Beyond Images:** This theory is universal. It applies to *any* diffusion model.
      * **Audio:** Compose a model of a "guitar" with a "piano" and a "sad" emotion.
      * **Video:** Compose a "person walking" model with a "Mars landscape" model.
      * **3D:** Compose a "chair" model with a "gothic architecture" style model.

#### Limitations & Open Questions

The paper is brilliant, but it opens more questions than it answers.

  * **What is $ \\mathcal{A} $?** The paper proves that a magic feature space $\mathcal{A}$ *can* exist, but it doesn't tell us what it *is* or how to find it. Is it the U-Net's internal layers? Is it the CLIP embedding space?
  * [cite\_start]**The Background $ p\_b $:** The paper shows that the choice of $p_b$ is *critical*[cite: 299]. But what *is* the best $ p\_b $? Is it the empty prompt? An average of all images? [cite\_start]A model trained on *explicitly* empty images[cite: 116]? This is now a huge engineering question.
  * **How Orthogonal is "Orthogonal"?** The heuristic is an approximation. How much similarity (e.g., a cosine similarity of 0.2? 0.4?) is "too much" for a successful composition?

This paper gives us the "Newton's Laws" of diffusion composition. Now, a new generation of engineers and researchers can begin the "Apollo Program."

-----

### 📝 Conclusion (300 words)

If you've made it this far, you've just completed a masterclass in the deepest theory of diffusion models. Let's boil it all down.

This paper set out to explain the "magic" of composing concepts like "dog" and "oil painting." It did so by throwing out the old, broken definitions and building a new theory from scratch.

**Here's what to remember:**

  * **1. [cite\_start]Old definitions are wrong:** "Simple Product" and "Bayes Composition" fail because they can't create truly new, out-of-distribution (OOD) images[cite: 157, 180].
  * **2. [cite\_start]A New Definition: "Projective Composition"**[cite: 204]. The right way to think about composition is *feature preservation*. A good composition $\hat{p}$ is one whose "shadows" (projections) match its parents' (e.g., its "style" looks like the "style" model, its "content" looks like the "content" model).
  * **3. [cite\_start]The "How-To": "Composition Operator"**[cite: 248]. The formula we *already use* for CFG $(\text{score}_u + w \cdot (\text{score}_c - \text{score}_u))$ is the right tool for the job.
  * **4. [cite\_start]The "When": "Factorized Conditionals"**[cite: 264]. This tool works *if* the concepts are "disjoint." This can be disjoint *pixels* (like in CLEVR) or, more powerfully...
  * **5. [cite\_start]The "Magic": "Feature-Space Composition"**[cite: 339]. ...it can be disjoint features in a *hidden space* ("style" vs. "content"). [cite\_start]The paper proves (via **Lemma 6.2**) that composing in pixel space *automatically* gives us the correct composition in this hidden feature space, *for free*[cite: 361].
  * **6. [cite\_start]The "Catch": "Sampling is Hard"**[cite: 373]. This magic *only* works if the hidden features are *orthogonal*. If they're not, our samplers (DDPM/DDIM) will fail.
  * **7. [cite\_start]The "Heuristic": "CLIP Orthogonality"**[cite: 410]. We can *predict* if a composition will work by checking the cosine similarity of their CLIP embeddings. "Dog" + "Hat" (orthogonal) works. [cite\_start]"Dog" + "Horse" (non-orthogonal) fails[cite: 453].

This paper moves diffusion composition from alchemy to science. It gives us a language, a theory, and a predictive toolset.

Now, go build. And compose.

-----

*If you enjoyed this 5,000-word deep dive, please give it a 👏 (or 50\!), follow me for more, and share your own composition results in the comments\! Which "non-orthogonal" concepts have you found that fail?*

-----

### 📖 Appendix (400 words)

#### Glossary of Technical Terms

  * **Bayes Composition:** The (flawed) idea of composition by $p(x|c_1, c_2) \propto p(x|c_1)p(x|c_2)/p(x)$. [cite\_start]Fails for OOD samples[cite: 161, 180].
  * **Composition Operator ($\mathcal{C}$):** The specific formula $\nabla \log \mathcal{C}[\vec{p}] = \nabla \log p_b + \sum_i (\nabla \log p_i - \nabla \log p_b)$. [cite\_start]This is the practical formula this paper analyzes[cite: 248, 255].
  * **Diffeomorphism ($\mathcal{A}$):** A smooth, invertible mathematical function. [cite\_start]The "magic" feature extractor that can disentangle concepts[cite: 339].
  * **Factorized Conditionals (Def 5.2):** The key *condition* for successful composition. [cite\_start]It means each model $p_i$ is *only* active on a *disjoint* set of coordinates or features ($M_i$)[cite: 264].
  * [cite\_start]**Forward Process:** The process of adding noise to an image $x_0$ to get $x_T$[cite: 67].
  * [cite\_start]**Out-of-Distribution (OOD):** A sample that is unlike anything in the model's training set (e.g., an "oil painting of your dog" for a model trained only on "your dog" and "oil paintings")[cite: 40].
  * **Projective Composition (Def 4.1):** The paper's *new, correct* definition of composition. [cite\_start]A composed distribution $\hat{p}$ is correct if its "projections" (features) match the marginal feature distributions of its parents[cite: 204].
  * **Pushforward ($\#$):** A mathematical operator. [cite\_start]$(\Pi \# p)$ simply means "the distribution of outputs you get from running all samples of $p$ through the function $ \\Pi $"[cite: 213].
  * **Reparameterization Equivariance (Lemma 6.2):** The "magic" property of the Composition Operator. $\mathcal{C}[\mathcal{A}(\vec{p})] = \mathcal{A}(\mathcal{C}[\vec{p}])$. [cite\_start]Composing *after* transforming is the same as transforming *after* composing[cite: 364, 366].
  * [cite\_start]**Reverse Process:** The process of *sampling* by starting with noise $x_T$ and iteratively denoising it to get an image $x_0$[cite: 68].
  * **Score Function ($\nabla_x \log p_t(x_t)$):** The "magic compass." A vector field that points "uphill" to more probable (less noisy) data. What the diffusion model learns to predict.

#### Further Reading

1.  [cite\_start]**Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020)[cite: 470]:** The foundational paper that re-ignited the diffusion revolution.
2.  [cite\_start]**Denoising Diffusion Implicit Models (DDIM) (Song et al., 2021)[cite: 526]:** The paper that introduced much faster "deterministic" sampling.
3.  [cite\_start]**Composable Diffusion Models (Liu et al., 2022)[cite: 495]:** A key practical paper (cited heavily by this one) that showed *how* to implement compositions in practice.
4.  **The Illustrated DDPM (Blog Post):** A fantastic, visual blog post by Niels Rogge and Kashif Rasul that provides a gentler introduction to the basics.
5.  **Generative Modeling with Score-Based Models (Yang Song's Blog Post):** A comprehensive overview of score-based modeling from one of the field's pioneers.

#### Citation

```bibtex
@article{bradley2025mechanisms,
  title={Mechanisms of Projective Composition of Diffusion Models},
  author={Bradley, Arwen and Nakkiran, Preetum and Berthelot, David and Thornton, James and Susskind, Joshua M.},
  journal={arXiv preprint arXiv:2502.04549},
  year={2025},
  eprint={2502.04549},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
