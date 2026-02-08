import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model

IMG_SIZE = 224

def focal_loss(alpha=0.65, gamma=1.5):
    """Focal loss for handling class imbalance."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_loss_value = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        
        return tf.reduce_mean(focal_loss_value)
    return loss_fn

def build_model(use_focal_loss=True, learning_rate=5e-4, dropout_rate=0.4, dense_units=256):
    """Building the ensemble model using EfficientNetB0 and ResNet50."""
    
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    
    
    # I am creating the EfficientNetB0 branch
    efficientnet_base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer,
        pooling=None
    )
    efficientnet_base.trainable = True
    for layer in efficientnet_base.layers[:-30]:
        layer.trainable = False
    
    efficientnet_features = GlobalAveragePooling2D(name='efficientnet_pool')(efficientnet_base.output)
    
    # Here we are creating the ResNet50 branch
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer,
        pooling=None
    )
    resnet_base.trainable = True
    for layer in resnet_base.layers[:-30]:
        layer.trainable = False
    
    resnet_features = GlobalAveragePooling2D(name='resnet_pool')(resnet_base.output)
    
    # I am concatenating features from both models
    
    concatenated = Concatenate(name='concat_features')([efficientnet_features, resnet_features])
    
    # Here we are building the classification head
    
    
    dense1 = Dense(dense_units, activation='relu', name='dense1')(concatenated)
    dropout = Dropout(dropout_rate, name='dropout')(dense1)
    output = Dense(1, activation='sigmoid', name='output')(dropout)
    
    model = Model(inputs=input_layer, outputs=output)
    
    loss_fn = focal_loss(alpha=0.65, gamma=1.5) if use_focal_loss else 'binary_crossentropy'
    
    
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model
