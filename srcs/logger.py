import tensorflow as tf
import numpy as np
from PIL import Image
import io

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert image to bytes
                s = io.BytesIO()
                Image.fromarray(img).save(s, format='PNG')
                img_tensor = tf.image.decode_png(s.getvalue(), channels=3)
                # Add batch dimension
                img_tensor = tf.expand_dims(img_tensor, 0)
                tf.summary.image(name=f'{tag}/{i}', data=img_tensor, step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)
            self.writer.flush()