
# Kindly download the model path from:
# http://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy
from BBox import bbobject as bb


im_path = 'BBox/poodle.png'
model_path = 'BBox/bvlc_alexnet.npy'
alex = bb.AlexNetClassifier(im_path, model_path)
alex.print_class([alex.im])
alex.nearest_crop()
