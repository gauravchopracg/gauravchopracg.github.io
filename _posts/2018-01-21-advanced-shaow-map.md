---
layout: post
title: "Advanced Shadow Mapping"
---

The previous [post](https://misaka-10032.github.io/shadow-map/) introduces the basics of shadow mapping. However, I didn't mention one of the fundamental problems: _aliasing_. If we take a closer look at the shadow, we will find it jagged on the edge.

![]({{ BASE_PATH }}/images/20180113/basic-shadow.png)

This is a common problem for pre-computed texels: if we had a depth map of infinite resolution, the problem would have been solved. The _shadow acne_ problem is also caused by this, though it could be hidden by applying a small bias. However, we have no luck for aliasing.

A lot of research has been done to mitigate aliasing, and to create soft shadow at the edge to look more realistic, such as Percentage Closer Filtering, Variance Shadow Maps, Moment Shadow Mapping.

## Percentage Closer Filtering

This is one of the most fundamental work. For simplicity, I'm only going to walk through a very basic version of it. The idea is intuitive: instead of sampling from a texel that is closest to the true coordinate, sample from a window (e.g. 2x2), and do an average (or weighted average, according to distance).

![]({{ BASE_PATH}}/images/20180121/pcf-2x2-window.png)

Remember the way we sample from the depth texture

```glsl
texture(uDepth, lightCoord.xy);
```

Instead of doing this, we sample from a window of texels. For clarity, let's say `uDepthMapScale` is passed in as uniform that is equal to `vec2(1./depthMapWidth, 1./depthMapHeight)`. Also we enlarge the window a little bit (4x4) to get fuzzier edges.

```glsl
float x, y, visibility = 0.;
for (y = -1.5; y <= 1.5; y += 1.) {
  for (x = -1.5; x <= 1.5; x += 1.) {
    float occluderLightDist =
        texture(uDepthMap, lightCoord.xy + uDepthMapScale * vec2(x, y)).z;
    visibility += weight(x, y) * float(fragLightDist < occluderLightDist + kEps);
  }
}
```

![]({{ BASE_PATH }}/images/20180121/pcf-shadow.png)

More formally, let's define $z_f$ as the distance from fragment to light and $z_o$ as the distance from occluder to light. Given a depth window in the view of light, the window of $z_o$'s, let's say $Z_o$, the visibility of the fragment could be modeled as

$$
V(z_f) = W * I(z_f < Z_o)
$$

where $W$ is the filter, $*$ is [convolution](https://en.wikipedia.org/wiki/Convolution), and $I$ is [indicator function](https://en.wikipedia.org/wiki/Indicator_function). For example, if we apply a 2x2 average filter on a depth window like this

$$
\begin{pmatrix}
0.2 & 0.2 \\
0.6 & 0.8
\end{pmatrix}
$$

then $V(z_f)$ would be a multi-step function

![]({{ BASE_PATH }}/images/20180121/multi-step.png)

For $z_f$ that is on the edge of a step, there still might be chance of shadow acne or aliasing.

### Variance Shadow Maps

Given a pre-computed depth map and a texture coordinate $(x, y)$, the $z_o$ sampled from it is actually an inaccurate one: some information has already been lost due to the finite rasterization. The true $z_o$ is actually a random variable that we don't know; the convolution we did in PCF is neverthless a good guess based on the continuity nature of the real scene.

However, in PCF we have to apply a convolution kernel per fragment in the shadow pass, which could be expensive if the kernel size is very large. Then someone asks the question: can we pre-compute the convolution in the depth pass, and only sample from one texel in the shadow pass? One of the benefits is that, if the light doesn't move, in other words, the pre-computed depth map doesn't change, the shadow pass could be much rendered much faster.

Actually, to sample $z_o$ from a window, we don't need all the values from it; some statistics about the window might be good enough. In [Variance Shadow Maps](http://www.punkuser.net/vsm/vsm_paper.pdf), the author proposes to collect the first two moments per window in the depth pass, and model the visibility as

$$
V(z_o, z_f) = P(z_o \ge z_f) \le \frac{\sigma^2}{\sigma^2 + (z_f - \mu)^2}
$$

The inequality is [Chebychev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality). $\mu$ and $\sigma$ could be pre-computed in the depth pass.

$$
\mu = E(z_o) \\
\sigma^2 = E(z_o^2) - E(z_o)^2
$$

$z_o$'s are sampled per window, so they could be pre-computed via convolution. For clarity, let's say the pre-computed result will have each texel being the weighted average of $(z, z^2, 0, 1)$ of a window around it. Given a texel sample, and the fragment-light distance, we can compute the visibility as following

```glsl
float computeVisibility(vec4 depthMapSample, float fragLightDist) {
  float mu = depthMapSample.x;
  if (mu > fragLightDist) {
    return 1.;
  }
  float mu2 = mu * mu;
  float sigma2 = clamp(depthMapSample.y - mu2, 1e-8, 1e8);
  float fragOccluderDist = fragLightDist - mu;
  float fragOccluderDist2 = fragOccluderDist * fragOccluderDist;
  return sigma2 / (sigma2 + fragOccluderDist2);
}
```

Notice that we applied a trick by early return at `mu > fragLightDist`. Otherwise, there will be very bad artifact at fragment where $z_f < z_o$ for real.

![]({{ BASE_PATH }}/images/20180121/bad-variance-shadow.png)

From math, we can easily analyze the side effect of the term $(z_f - \mu)^2$, so to avoid the risk, we simply ignore this term when $z_f < \mu$.

![]({{ BASE_PATH }}/images/20180121/variance-shadow.png)

This method is not perfect neither. Notice that we modeled the visibility as upper bound of $P(z_o \ge z_f)$. The upper bound may not be tight, so there's chance that the visibility is greater than its true value. The consequence is that some region will be brighter than what it should be, and this artifact is known as _Light Bleeding_.

### Moment Shadow Maps

Is it possible to get a closer upper bound given more moments? The answer is yes. In [Moment Shadow Maps](http://momentsingraphics.de/?page_id=51), the author generalizes the problem as [Hamburger Moment Problem](https://en.wikipedia.org/wiki/Hamburger_moment_problem), and gives the following solution with 4 moments.

* Let $b = (z, z^2, z^3, z^4)$
* Let $b' = (1 - \alpha) \cdot b + \alpha \cdot (0., .63, 0., .63)^T$, $\alpha = 9 \cdot 10^{-3}$
* Solve $c$ with Cholesky decomposition

$$
\begin{pmatrix}
  1 & b_1' & b_2' \\
  b_1' & b_2' & b_3' \\
  b_2' & b_3' & b_4'
\end{pmatrix}
\cdot c =
\begin{pmatrix}
  1 \\ z_f \\ z_f^2
\end{pmatrix}
$$

* Solve $d_1 \le d_2$ for

$$
c_3 \cdot d^2 + c_2 \cdot d + c_1 = 0
$$

* Compute visibility given $z_f$, $b'$, $d$.

$$
V = \begin{cases}
  1 &, z_f \le d_1 \\
  1 - \frac{z_f d_2 - b_1'(z_f + d_2) + b_2'}{(d_2 - d_1)(z_f - d_1)} &, d_1 < z_f \le d_2 \\
  \frac{d_1 d_2 - b_1' (d_1 + d_2) + b_2'}{(z_f - d_1) (z_f - d_2)} &, z_f > d_2
\end{cases}
$$

```glsl
float computeVisibility(vec4 depthMapSample, float fragLightDist) {
  float alpha = 9e-3;
  vec4 b = (1. - alpha) * depthMapSample + alpha * vec4(0., .63, 0, .63);
  mat3 B = mat3(
      1.0, b.x, b.y,
      b.x, b.y, b.z,
      b.y, b.z, b.w);
  float zf = fragLightDist;
  vec3 z = vec3(1, zf, zf * zf);
  vec3 c = inverse(B) * z;
  float sqrtDelta = sqrt(c.y * c.y - 4. * c.z * c.x);
  float d1 = (-c.y - sqrtDelta) / (2. * c.z);
  float d2 = (-c.y + sqrtDelta) / (2. * c.z);
  if (d2 < d1) {
    float tmp = d1; d1 = d2; d2 = tmp;
  }
  if (zf <= d1) {
    return 1.;
  } else if (zf <= d2) {
    return 1. - (zf * d2 - b.x * (zf + d2) + b.y) / ((d2 - d1) * (zf - d1));
  } else {
    return (d1 * d2 - b.x * (d1 + d2) + b.y) / ((zf - d1) * (zf - d2));
  }
}
```

![]({{ BASE_PATH }}/images/20180121/moment-shadow.png)

### Quantization

TODO

## Reference

* [Tutorial](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/) on shadow mapping.
* GPU Gems [tutorial](https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch11.html) on Percentage Closer Filtering.
* [Paper](http://www.punkuser.net/vsm/vsm_paper.pdf) on Variance Shadow Maps.
* [Lecture](http://momentsingraphics.de/?page_id=51) on Moment Shadow Mapping.
