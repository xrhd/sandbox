from metaflow import FlowSpec, step

class MNISTFlow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        import tensorflow_datasets as tfds  # TFDS to download MNIST.
        import tensorflow as tf  # TensorFlow / `tf.data` operations.

        tf.random.set_seed(0)  # Set the random seed for reproducibility.

        train_steps = 1200
        eval_every = 200
        batch_size = 32

        train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
        test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

        train_ds = train_ds.map(
          lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
          }
        )  # normalize train set
        test_ds = test_ds.map(
          lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
          }
        )  # Normalize the test set.

        # Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
        self.train_ds = train_ds.repeat().shuffle(1024)
        # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
        self.train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
        # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
        self.test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

        self.next(self.end)

    @step
    def end(self):
        print("Treinamento finalizado!")

if __name__ == "__main__":
    MNISTFlow()

