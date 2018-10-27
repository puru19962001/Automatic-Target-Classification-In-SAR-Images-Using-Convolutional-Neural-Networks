
from scipy import misc
import tensorflow as tf
import numpy as np
import os,glob
import pandas as pd
            

# First, pass the path of the image
path = 'C:\\Users\\RAD5\\Desktop\\Student Resources New\\MSTAR\\Testing Data'
size_image=64
num_channels=1
images = []
classes = ['BTR_60','2S1','BRDM_2','D7','SLICY','T62','ZIL131','ZSU_23_4']
target_class = []
#im = misc.imread(fl)
for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.abspath('C:\\Users\\RAD5\\Desktop\\Student Resources New\\MSTAR\\Testing Data')
        
        
        for root, dirs, files in os.walk(path, topdown=True):
            for direct in dirs:
                if direct == fields:
                    path=os.path.join(root, direct, '*.jpg')
         
                    files1 = glob.glob(path) 
                    for fl in files1:
                        im = misc.imread(fl)
                        im = misc.imresize(im, [size_image, size_image])
                        im = np.atleast_3d(im)
                        images.append(im)
                        target_class.append(fields)
                        #print(fl)


images = np.array(images)
target_class = np.array(target_class)
print("Number of files in Testing-set:\t\t{}".format(images.shape[0]))

#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
#x_batch = images.reshape(-1, size_image,size_image,num_channels)
x_batch = images
## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('mstar-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1,8))



### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
tgt_class=sess.run(tf.argmax(result,1), feed_dict=feed_dict_testing)


print(target_class)
df2 = pd.DataFrame(target_class)
df2.to_csv('TargetClassNames.csv');

print(result)
df = pd.DataFrame(result)
df.to_csv('Results.csv');

df1 = pd.DataFrame(tgt_class)
df1.to_csv('ClassesIndices.csv');
