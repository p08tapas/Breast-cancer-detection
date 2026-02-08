import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Sequential


IMG_SIZE = 224  # I use the standard size for ResNet50


def focal_loss(alpha=0.65, gamma=1.5):
    """
    Moderate alpha (0.65) value has been chosen to balance accuracy and sensitivity.This needs to be experimented with
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
          
        # Calculating binary crossentropy
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        
             
        # Here we are down-weighting easy examples with focal weight
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_loss_value = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        
        return tf.reduce_mean(focal_loss_value)
    return loss_fn


def build_model(use_focal_loss=True):
    """I am building the transfer learning model using ResNet50 with ImageNet weights."""
    
    # Loading ResNet50 with ImageNet pre-trained weights
    

    base = ResNet50(
        include_top=False,
        weights='imagenet',  # I am using ImageNet pre-trained weights
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ) 
   
 
    # Here I am fine-tuning top layers for model customization
    # Freezing bottom layers and keeping top layers trainable for efficient fine-tuning
    base.trainable = True
    # Freezing all but the last 30 layers for efficient fine-tuning
    for layer in base.layers[:-30]:
        layer.trainable = False
    
  
    # Building the model with an improved classification head for better performance
    # Increased capacity: 512 -> 256 -> 1 with BatchNorm and Dropout
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
   
    # I use balanced focal loss or binary crossentropy
    loss_fn = focal_loss(alpha=0.65, gamma=1.5) if use_focal_loss else 'binary_crossentropy'
    
    # Using AdamW optimizer with weight decay for better generalization
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss_fn,
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

