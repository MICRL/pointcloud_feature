#!/usr/bin/env python

import rospy
from keras.models import Sequential, model_from_json
import numpy as np
from pointcloud_feature.srv import *

model = None

def handle_feature_extract(req):
    result = model.predict(2.0 * np.asarray(req.grid).reshape(1,1,32,32,32) - 1.0)
    return FeatureExtractResponse(list(result.reshape(np.prod(result.shape))))

def feature_extract_server():
    global model
    rospy.init_node("feature_extract_server")

    # Load the model
    with open('model_config.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights("model_weights.h5")

    s = rospy.Service('feature_extract', FeatureExtract, handle_feature_extract)
    print "Ready to extract feature."
    rospy.spin()

if __name__ == "__main__":
    feature_extract_server()
