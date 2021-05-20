#!/usr/bin/env python
# coding: utf-8

# In[49]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import seaborn as sns
sns.set()


# In[50]:


import keras


# In[51]:


data_dir='E:\cricket\cricket_players'


# In[52]:


df=pd.read_csv('E:\Internship\players.csv')


# In[53]:


df=df.drop('Unnamed: 0',axis=1)


# In[54]:


df.head(5)


# In[55]:


translate = {"bhuvneshwar_kumar": "bhuvneshwar_kumar",
"dinesh_karthik":"dinesh_karthik",
"hardik_pandya": "hardik_pandya",
"jasprit_bumrah": "jasprit_bumrah",
"k._l._rahul": "k._l._rahul",
"kedar_jadhav": "kedar_jadhav",
"kuldeep_yadav":"kuldeep_yadav",
"mohammed_shami": "mohammed_shami",
"ms_dhoni": "ms_dhoni",
"ravindra_jadeja":"ravindra_jadeja",
"rohit_sharma":"rohit_sharma",
"shikhar_dhawan": "shikhar_dhawan",
"vijay_shankar":"vijay_shankar",
"virat_kohli":"virat_kohli",
"yuzvendra_chahal":"yuzvendra_chahal"}


# "bhuvneshwar_kumar": "bhuvneshwar_kumar",
# "dinesh_karthik":"dinesh_karthik",
# "hardik_pandya": "hardik_pandya",
# "jasprit_bumrah": "jasprit_bumrah",
# "k._l._rahul": "k._l._rahul",
# "kedar_jadhav": "kedar_jadhav",
# "kuldeep_yadav":"kuldeep_yadav",
# "mohammed_shami": "mohammed_shami",
# "ms_dhoni": "ms_dhoni",
# "ravindra_jadeja":"ravindra_jadeja",
# "rohit_sharma":"rohit_sharma",
# "shikhar_dhawan": "shikhar_dhawan",
# "vijay_shankar":"vijay_shankar",
# "virat_kohli":"virat_kohli",
# "yuzvendra_chahal":"yuzvendra_chahal"

# In[56]:


plt.figure(figsize=(24,3))
sns.countplot(df['player'])


# In[57]:


foldernames = os.listdir('E:\cricket\cricket_players')
files, files2, target, target2 = [], [], [], []

#Iterate through the database and retrieve our relevant files
for i, folder in enumerate(foldernames):
    filenames = os.listdir('E:\cricket\cricket_players' + '\\' + folder)
    count = 0
    #Due to the specific nature of the database being used, there are 1446 images of a specific class (others are higher)
    #Hence use a maximum of 1400 images from a specific classes for consistency of data as well as brevity
    for file in filenames:
        if count < 21:
            # yaha \\ laga raha hu
            files.append('E:\cricket\cricket_players' + '\\'+folder + "\\" + file)
            target.append(translate[folder])
        else:
            files2.append('E:\cricket\cricket_players' +'\\' +folder + "\\" + file)
            target2.append(translate[folder])
        count += 1

#Create dataframes to read the images 
df1 = pd.DataFrame({'Filepath':files, 'Target':target})


# In[58]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[59]:


image_gen = ImageDataGenerator(rotation_range=20,
                               samplewise_center=True,
                               width_shift_range=0.10,
                               height_shift_range=0.10,
                               rescale=1/255,
                               shear_range=0.1, 
                               zoom_range=0.2,
                               horizontal_flip=True, 
                               vertical_flip=False,
                               fill_mode='nearest' 
                              )


# In[60]:


augdata_test = ImageDataGenerator(rescale=1./255, samplewise_center = True)


# In[66]:


df1=df1.sample(frac=1)


# In[67]:


train, vald, test = np.split(df1.sample(frac=1), [int(.8*len(df1)), int(.9*len(df1))])


# In[68]:


train_flow = image_gen.flow_from_dataframe(train, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
dev_flow = augdata_test.flow_from_dataframe(vald, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)
test_flow = augdata_test.flow_from_dataframe(test, x_col = 'Filepath', y_col = 'Target', target_size=(224, 224), interpolation = 'lanczos', validate_filenames = False)


# In[69]:


from keras import applications
vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[70]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense,Dropout,MaxPooling2D


# In[71]:


class_model = Sequential()
class_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
class_model.add(Dropout(0.2))


# In[72]:


class_model.add(Dense(128,activation='relu'))
class_model.add(Dropout(0.5))
class_model.add(Dense(15,activation='softmax'))


# In[73]:


from keras.models import Model


# In[74]:


model = Model(inputs=vgg16_model.inputs, outputs=class_model(vgg16_model.output)) 
model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])


# In[75]:


model.summary()


# In[76]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',patience=2,mode='auto')


# In[77]:


import warnings
warnings.filterwarnings('ignore')


# In[78]:


from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 1, verbose=1,factor=0.2, min_delta=0.0001, min_lr=0.00000001)


# In[85]:


results = model.fit_generator(train_flow,epochs=6,
                              validation_data=dev_flow,
                             callbacks=[ModelCheckpoint('VGG16.model', monitor='val_acc'), learning_rate_reduction])


# In[86]:


losses=pd.DataFrame(model.history.history)


# In[91]:


losses[['loss','val_loss']].plot()


# In[92]:


model.metrics_names


# In[93]:


score=model.evaluate(test_flow)


# In[94]:


score


# In[95]:


pred_probabilities=model.predict_generator(test_flow)


# In[96]:


pred=model.predict(test_flow)


# In[97]:


from tensorflow.keras.models import load_model
model.save('cricket_player.h5')


# In[98]:


model_json = model.to_json()
with open("cricket_player.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('cricket_players.h5')


# In[ ]:




