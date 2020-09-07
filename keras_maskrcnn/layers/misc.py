"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np


class Shape(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.shape(inputs)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)


class ConcatenateBoxes(layers.Layer):
    def call(self, inputs, **kwargs):
        boxes, other = inputs

        boxes_shape = K.shape(boxes)
        other_shape = K.shape(other)
        other = K.reshape(other, (boxes_shape[0], boxes_shape[1], -1))

        return K.concatenate([boxes, other], axis=2)

    def compute_output_shape(self, input_shape):
        boxes_shape, other_shape = input_shape
        return boxes_shape[:2] + (np.prod([s for s in other_shape[2:]]) + 4,)


class Cast(layers.Layer):
    def __init__(self, dtype=None, *args, **kwargs):
        if dtype is None:
            dtype = K.floatx()
        self.dtype = dtype

        super(Cast, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        outputs = K.cast(inputs, self.dtype)
        return outputs
