import tensorflow as tf
from models import Introspector


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate=2e-5, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5) + self.initial_learning_rate

        return tf.math.minimum(arg1, arg2)


class IntrospectorModule:
    def __init__(self, input_shape, epochs=3, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        train_data_size = input_shape[0]

        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        learning_rate = CustomSchedule(2e-5)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        self.introspector = Introspector(input_shape[-1])

    def train_step(self, input):
        input_ids = input[:, 0, :]
        attention_mask = input[:, 1, :]
        y_real = input[:, 3, :][:, :, tf.newaxis]
        with tf.GradientTape() as tape:
            y_pred = self.introspector({
                'input_ids': input_ids, 'attention_mask': attention_mask})
            loss = self.loss_fn(y_real, y_pred)
        gradients = tape.gradient(loss, self.introspector.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.introspector.trainable_variables))
        return loss

    def training_epoch(self, inputs):
        datasetes = tf.data.Dataset.from_tensor_slices(inputs).batch(self.batch_size)
        for epoch in range(self.epochs):
            for index, dataset in enumerate(datasetes):
                print(f"\033[0;35mIndex:\033[0;36m{index}\033[0m")
                loss = self.train_step(dataset)
                print(f"\033[0;33mLoss:\033[0;34m{loss}\033[0m")

    def training_fit(self, inputs):
        self.introspector.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy'])

        self.introspector.fit(
            x={'input_ids': inputs[:, 0, :], 'attention_mask': inputs[:, 1, :]},
            y=inputs[:, 3, :][:, :, tf.newaxis], batch_size=self.batch_size, epochs=self.epochs)
