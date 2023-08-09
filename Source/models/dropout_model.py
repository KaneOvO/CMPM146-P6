from models.model import Model
import tensorflow as tf
from tensorflow.keras import layers, models

class DropoutModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        # self.model = <model definition>
        #pass
        self.model = models.Sequential()

        self.model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
        
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        

        self.model.add(layers.Flatten())

        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(64, activation='relu'))  
        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        # Your code goes here

        # self.model.compile(<configuration properties>)
        #pass
        self.model.compile(
            optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.001),
            # optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )