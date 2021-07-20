

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# load model
model = load_model('Model.h5')

# summarize model
#model.summary()

test_image = image.load_img(r"E:\Deep learning\Task\My Task in ML&DL\Custom_testing_images\app.png", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'Not an Apple'
    print({ "image" : prediction})
else:
    prediction = 'Apple'
    print({ "image" : prediction})
