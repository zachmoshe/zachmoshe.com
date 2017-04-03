---
layout: post
title: "Pickling Keras Models"
image: /images/articles/pickling-keras-models/image.png
tags: python deep-learning keras pickle
description: >
    A short post on how to serialize Keras Model objects with python's Pickle library
---

It's pretty annoying that Keras doesn't support Pickle to serialize its objects (Models). Yes, the Model structure is serializable (`keras.models.model_from_json`) and so are the weights (`model.get_weights`), and we can always use the built-in `keras.models.save_model` to store it as an hdf5 file, but all these won't help when we want to store another object that references the model (like `keras.callbacks.History`), or use the `%store` magic of iPython notebook.

After some frustration, I ended up with a patchy solution that does the work for me. It's not the nicest thing, but works regardless of how you reference your Keras model. Basically, if an object has `__getstate__` and `__setstate__` methods, [pickle will use them](https://docs.python.org/3/library/pickle.html#pickle-inst) to serialize the object. The problem is that Keras Model [doesn't implement these](https://github.com/fchollet/keras/issues/789). My patchy solution adds those methods after the class has been loaded:

{% highlight python %}
import types
import tempfile
import keras.models

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

{% endhighlight %}

And a usage example:

{% highlight python %}
import keras
import pickle

make_keras_picklable()

m = keras.models.Sequential()
m.add(keras.layers.Dense(10, input_shape=(10,)))
m.compile(optimizer='sgd', loss='mse')

pickle.dumps(m)

>> b'\x80\x03ckeras.models\nSequential\nq\x00)\x81q\x01}q\x02X...
{% endhighlight %}

I have a general python module that I always import on all my notebooks and contains some stuff I always need so I just added it there. You can just add to one of your initializers files or the beginning of your script.
