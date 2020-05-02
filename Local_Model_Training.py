#/bin/env python3
from keras.preprocessing.image import ImageDataGenerator
import keras

#import models
import Keras_Models as KM

augmented_data_directory = './aug_training/'
val_data = './val_data/'
model_storage = './best_model.mdl'

if __name__=="__main__":
    #create image data generators
    training_data_iterator = ImageDataGenerator().flow_from_directory(augmented_data_directory, target_size=(480,640))
    validation_data_iterator = ImageDataGenerator(rescale=1./255).flow_from_directory(val_data, target_size=(480,640))

    model = KM.generate_base_model((480,640, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        './best_model.mdl', 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False, 
        mode='auto', 
        period=1
    )

    history = model.fit_generator(
        generator=training_data_iterator, 
        steps_per_epoch=512, 
        epochs=16, 
        validation_data=validation_data_iterator,
        callbacks=[checkpoint_callback],
        use_multiprocessing=True,
        workers=2
    )

    # Plot training & validation loss values
    fig=plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_loss_1.fig', format='svg')
    
    fig2 = figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_accuracy_1.fig', format='svg')