{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "from tensorflow.python.keras.layers import Convolution2D\n",
    "from tensorflow.python.keras.layers import MaxPooling2D\n",
    "from tensorflow.python.keras.layers import Dense\n",
    "from tensorflow.python.keras.layers import Flatten\n",
    "from tensorflow.python.keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_2 (Conv2D)            (None, 62, 62, 64)        1792      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 31, 31, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 29, 29, 32)        18464     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 14, 14, 32)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 6272)              0         \n",
      "_________________________________________________________________\n",
      "dense_6 (Dense)              (None, 128)               802944    \n",
      "_________________________________________________________________\n",
      "dense_7 (Dense)              (None, 64)                8256      \n",
      "_________________________________________________________________\n",
      "dense_8 (Dense)              (None, 32)                2080      \n",
      "_________________________________________________________________\n",
      "dense_9 (Dense)              (None, 16)                528       \n",
      "_________________________________________________________________\n",
      "dense_10 (Dense)             (None, 8)                 136       \n",
      "=================================================================\n",
      "Total params: 834,200\n",
      "Trainable params: 834,200\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model=Sequential()\n",
    "model.add(Convolution2D(filters=64 , kernel_size=(3,3) , activation='relu' , input_shape=(64,64,3)))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Convolution2D(filters=32 , kernel_size=(3,3) , activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(units=128 , activation='relu'))\n",
    "model.add(Dense(units=64 , activation='relu'))\n",
    "model.add(Dense(units=32 , activation='relu'))\n",
    "model.add(Dense(units=16 , activation='relu'))\n",
    "model.add(Dense(units=8, activation='relu'))\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from tensorflow.python.keras.optimizers import Adam\n",
    "model.compile(optimizer='Adam' , loss='binary_crossentropy' , metrics=['accuracy'])\n",
    "from keras_preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.python.keras.preprocessing import image\n",
    "model.add(Dense(units=1 , activation='sigmoid'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "        rescale=1./255,\n",
    "        shear_range=0.2,\n",
    "        zoom_range=0.2,\n",
    "        horizontal_flip=True)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 8000 images belonging to 2 classes.\n",
      "Found 2000 images belonging to 2 classes.\n",
      "  36/8000 [..............................] - ETA: 2:34:37 - loss: 0.6941 - acc: 0.4936"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-28-c7e57ed61ca3>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     14\u001b[0m         \u001b[0mepochs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     15\u001b[0m         \u001b[0mvalidation_data\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtest_set\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 16\u001b[1;33m         validation_steps=800)\n\u001b[0m",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)\u001b[0m\n\u001b[0;32m    817\u001b[0m         \u001b[0mmax_queue_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmax_queue_size\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    818\u001b[0m         \u001b[0mworkers\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mworkers\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 819\u001b[1;33m         use_multiprocessing=use_multiprocessing)\n\u001b[0m\u001b[0;32m    820\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    821\u001b[0m   def evaluate(self,\n",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training_generator.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[0;32m    602\u001b[0m         \u001b[0mshuffle\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mshuffle\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    603\u001b[0m         \u001b[0minitial_epoch\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0minitial_epoch\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 604\u001b[1;33m         steps_name='steps_per_epoch')\n\u001b[0m\u001b[0;32m    605\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    606\u001b[0m   def evaluate(self,\n",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training_generator.py\u001b[0m in \u001b[0;36mmodel_iteration\u001b[1;34m(model, data, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, validation_freq, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch, mode, batch_size, steps_name, **kwargs)\u001b[0m\n\u001b[0;32m    263\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    264\u001b[0m       \u001b[0mis_deferred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_is_compiled\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 265\u001b[1;33m       \u001b[0mbatch_outs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mbatch_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0mbatch_data\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    266\u001b[0m       \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch_outs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    267\u001b[0m         \u001b[0mbatch_outs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mbatch_outs\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mtrain_on_batch\u001b[1;34m(self, x, y, sample_weight, class_weight, reset_metrics)\u001b[0m\n\u001b[0;32m   1121\u001b[0m       \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_update_sample_weight_modes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msample_weights\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msample_weights\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1122\u001b[0m       \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_train_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1123\u001b[1;33m       \u001b[0moutputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtrain_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mins\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# pylint: disable=not-callable\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1124\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1125\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mreset_metrics\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\keras\\backend.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m   3565\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3566\u001b[0m     fetched = self._callable_fn(*array_vals,\n\u001b[1;32m-> 3567\u001b[1;33m                                 run_metadata=self.run_metadata)\n\u001b[0m\u001b[0;32m   3568\u001b[0m     \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call_fetch_callbacks\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfetched\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_fetches\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3569\u001b[0m     output_structure = nest.pack_sequence_as(\n",
      "\u001b[1;32mc:\\users\\admin\\.conda\\envs\\mlops\\lib\\site-packages\\tensorflow_core\\python\\client\\session.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1472\u001b[0m         ret = tf_session.TF_SessionRunCallable(self._session._session,\n\u001b[0;32m   1473\u001b[0m                                                \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_handle\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1474\u001b[1;33m                                                run_metadata_ptr)\n\u001b[0m\u001b[0;32m   1475\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mrun_metadata\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1476\u001b[0m           \u001b[0mproto_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf_session\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTF_GetBuffer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrun_metadata_ptr\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "\n",
    "training_set = train_datagen.flow_from_directory(\n",
    "        'training_set',\n",
    "        target_size=(64, 64),\n",
    "        batch_size=32,\n",
    "        class_mode='binary')\n",
    "test_set = test_datagen.flow_from_directory(\n",
    "        'test_set',\n",
    "        target_size=(64, 64),\n",
    "        batch_size=32,\n",
    "        class_mode='binary')\n",
    "model.fit(\n",
    "        training_set,\n",
    "        steps_per_epoch=8000,\n",
    "        epochs=1,\n",
    "        validation_data=test_set,\n",
    "        validation_steps=800)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
