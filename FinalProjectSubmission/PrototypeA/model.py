import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential


IMG_SIZE = 224  # I am using the EfficientNetB0 standard size



def focal_loss(alpha=0.65, gamma=1.5):
    """
    Moderate alpha (0.65) value has been chosen to balance accuracy and sensitivity.This needs to be experimented with
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
          
        # I am calculating the binary crossentropy first
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        
             
        # Here we are down-weighting easy examples with focal weight
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_loss_value = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        
        return tf.reduce_mean(focal_loss_value)
    return loss_fn




def build_model(use_focal_loss=True):
    """This is about building the transfer learning model using EfficientNetB0."""
    # First, we are loading EfficientNetB0 with ImageNet weights
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ) 
   
 
    # Now we arefine-tuning top layers for model customization
    # I have frozen the bottom layers and kept the top layers trainable
    base.trainable = True
    for layer in base.layers[:-30]:  # I freeze all but the last 30 layers
        layer.trainable = False
    
  
    # Building the complete model by stacking the base backbone with the classification head
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
   
    # Using balanced focal loss or binary crossentropy
    loss_fn = focal_loss(alpha=0.65, gamma=1.5) if use_focal_loss else 'binary_crossentropy'
    
    # Compiling the model with a lower learning rate for fine-tuning
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # Lower LR for fine-tuning
        loss=loss_fn,
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

