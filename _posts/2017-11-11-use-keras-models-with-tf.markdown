---
layout: post
title: Use Keras Pre-Trained Models With Tensorflow
image: /images/articles/use-keras-models-with-tf/image.png
tags: keras tensorflow vgg vgg19
description: >
     Some useful tips for using Keras pre-trained models (keras.applications) in
     your own Tensorflow graphs.
---

In my last post (the [Simpsons Detector](/2017/05/03/simpsons-detector.html)) I've
used Keras as my deep-learning package to train and run CNN models. Since Keras is just
an API on top of TensorFlow I wanted to play with the underlying layer and therefore implemented
[image-style-transfer](https://pdfs.semanticscholar.org/7568/d13a82f7afa4be79f09c295940e48ec6db89.pdf)
with TF.

Image-style-transfer requires calculation of `VGG19`'s output on the given images and
since I was familiar with the nice API of `Keras` and `keras.applications`, I expected that to work easily.

Well, that's not quite the case... While I could 'get things to work', I was always
confused by inconsistent behavior, weird occasional errors and messy graphs that made me
shamefully admit that I don't really understand what's going on.

After spending some time on that, here are 4 tips that I think will make your life
easier if you plan to use `Keras` pre-trained models in your `TensorFlow` graphs.
I also created my own [wrapper to `VGG19`](#my-vgg19-wrapper) to demonstrate that. Feel free to use as it is
or adjust to your needs.


## Keras Pre-Trained Models

`Keras` comes with some built-in models that implement famous widely-used applications with  
their pre-trained weights (on common datasets). This allows you to get results pretty fast and easy:

{% highlight python %}
vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False)
imgs = ...  # load images
imgs = ...  # apply VGG preprocessing
keras_output = vgg19.predict(imgs)
{% endhighlight %}

The first section in this [notebook](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb) runs this code on a sample
image I took a couple of years ago in New-Zealand. I'm using the `mean()` of the
activation map on the last VGG19 layer as a hash for the calculation results. We'll
compare that later with a second more TF-ish implementation.

{% highlight python %}
keras_output.shape, keras_output.mean()
# => ((1, 6, 9, 512), 1.5227494)
{% endhighlight %}


## Problems With Keras-TensorFlow Integration

Why would I even want to take a model from one package and run it in another?
I guess there could be many reasons for that, including some psychotic disorders,
but my use-case is much simpler - I wanted to implement an `image-style-transfer`
model and for that I needed to compute `VGG19` outputs on 3 images.
The model I needed is not a straight-forward fit/predict model, so I can't build
it with `Keras` only, but on the other hand, I don't really want to start building
in TF the full network of VGG and having to deal with loading weights.

I was naive at first, and expected something similar to [the functional API of `Keras`]
(https://keras.io/getting-started/functional-api-guide/)
to just work.  
__THIS DOESN'T WORK__:

{% highlight python %}
input_img = tf.placeholder(tf.float32, (1,200,300,3), name='input_img')
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
output = vgg19(input_img)

img = ... # load and preprocess image
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_val = sess.run(output, {input_img: img})

output_val.shape, output_val.mean()
# => ((1, 6, 9, 512), 0.067449108)
{% endhighlight %}

There are a few problems with this code but most eye-catching one is the fact that
the `mean()` of the activation map is not the same like in the 'pure' `Keras` code
from before.

Here are the obvious and hidden problems with just 'plain-integrating'
`Keras` models into `TensorFlow` code:

### 1. Using the model in a new session

Apparently, as anyone would notice after the first couple of minutes of playing
with this code, after we create the `VGG` model, we can't use it in a different
session (like in `with tf.Session() as sess: ...`). Here is a code to demonstrate
that:

{% highlight python %}
# Let's create the model like before:
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
print(vgg19.get_layer('block1_conv1').get_weights()[0].shape)
print("Everything was loaded as we would expect")

# And now, let's use it in a new session
with tf.Session().as_default():
    try:
        print(vgg19.get_layer('block1_conv1').get_weights()[0].shape)
    except Exception as ex:
        print("EXCEPTION: ", ex)

# => (3, 3, 3, 64)
# => Everything was loaded as we would expect
# => EXCEPTION:  Attempting to use uninitialized value block1_conv1_4/bias 	[[Node: _retval_block1_conv1_4/bias_0_0 = _Retval[T=DT_FLOAT, index=0, _device="/job:localhost/replica:0/task:0/cpu:0"](block1_conv1_4/bias)]]
{% endhighlight %}

It's pretty common to create a graph once and run it in many sessions, but here,
even with a simple use-case we get a weird error. When `Keras` loads our model with
pre-trained weights, it actually runs an `tf.assign` operation to set the values to
all the weights in the graph. Once we use a new session, this initialization is
gone and `TensorFlow` is left with uninitialized nodes.

A possible solution would be to create the model in the same session that we're
using it in (or pass a reference to that session), but that is not always possible.
Another solution is to use `model.load_weights(...)` in the new session.

[My wrapper for `VGG`](#my-vgg19-wrapper) (shown at the end) uses something similar to the `load_weights()`
approach.


### 2. tf.global_variables_initializer() will destroy pre-trained weights

Although implied from the previous section, it's important to understand that
your weights are variables and will be randomly initialized when calling the
global initializer. So even if you kept the session, but then called `tf.global_variables_initializer()`
to initialize your other variables - congratulations! you now have a random `VGG`
model.

[The notebook that follows this post](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb) shows exactly that. I won't bring the code
here to keep it shorter.




### 3. Graphs are created multiple times

Things might work after you understand the first 2 issues, but when you open
`tensorboard` and look on the graph, you'll see it's not as nice as you'd expect.
In the following example, I'm using VGG once to compute `output` and threfore
expect to see only one 'VGG block' in my graph. Instead it looks duplicated:

{% highlight python %}
img = tf.placeholder(tf.float32, (1,224,224,3), name='input_image')
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
output = vgg19(img)
{% endhighlight %}

![TensorFlow duplicate graph](/content/use-keras-models-with-tf/tensorflow-dup-graph.png)

The cause here is completely my fault, but a one I believe is easy to miss given the
`Keras` functional API. When I'm instantiating VGG19, it builds a graph. Then, when
I'm applying it on the input tensor, it builds another graph that is connected to
that input. The first graph was never used and therefore is not connected to anything
(Keras created a new input tensor for it). It's basically just some garbage in the graph.

The solution is to use `input_tensor=input` parameter to the VGG constructor instead
of the (confusing) Keras way of `vgg19(input)`.


### 4. Model weights are trainable

Another one that is implied from before but easy to miss due to Keras API is the
fact model weights will also be trained (unless specifically excluded).
Notice that the `trainable` attribute of the `Keras` Model has no effect as we're
not compiling the model with `Keras`.

Like in previous sections, the [notebook](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb) shows an example
that 'proves' this. I've used the sum of a specific layer weights and the sum of
the image variable as indicators to whether they're changing or not.

In order to handle this, I've added to [my Keras wrapper](#my-vgg19-wrapper) the `model_weights_tensors`
attribute that returns a set of the VGG weights tensors so you can exclude them
from training. A full example is in the notebook, but basically you have to use
`optimizer.minimize(..., var_list=VARS_TO_TRAIN)`.

## My VGG19 Wrapper

In order to address all these, and have a re-usable component that I can actually
work with, I've wrapped VGG19 with my own short class.
Feel free to use or adjust to your needs.

[Code is available here](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/vgg19.py)
and also attached to the [notebook](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb).

Here is what it basically does:

  * Can be initialized with an input_tensor (otherwise, a placeholder will be created and stored in `self.input_tensor`)
  * Deals with VGG preprocessing (subtract VGG_MEAN and flips RGB to BGR)
  * Creates a clean graph. Different parts has different name scopes
  * Saves a checkpoint from the session used when loading the model with the
  pre-trained weights. Exposes a `load_weights()` method to restore weights from
  checkpoint
  * Expose all layers' outputs with `__getitem__` access (`vgg['block5_pool']` for
  example)

And here is a short example (also demonstrated in the [notebook](https://github.com/zachmoshe/zachmoshe.com/blob/master/content/use-keras-models-with-tf/using-keras-models-in-tf.ipynb))
and the `TensorFlow` graph it generates:

{% highlight python %}
from image_style_transfer import VGG19
IMAGE_SHAPE = (1,200,300,3)

my_img = tf.placeholder(tf.float32, IMAGE_SHAPE, name='my_original_image')
vgg = VGG19(image_shape=IMAGE_SHAPE, input_tensor=my_img)

output = tf.identity(vgg['block5_pool'], name='my_output')  # just to create an 'output' node in the graph
{% endhighlight %}

![My VGG19 Graph](/content/use-keras-models-with-tf/my-vgg-graph.png)

Just for comparison, we can calculate the mean output of `block5_pool` and compare
to the 'pure' `Keras` approach:

{% highlight python %}
imgs = ...    # load images
with tf.Session() as sess:
    vgg.load_weights()

    output_val = sess.run(output, { my_img: imgs })

output_val.shape, output_val.mean()
# => (1, 6, 9, 512) 1.52275
{% endhighlight %}

Exactly the same!
