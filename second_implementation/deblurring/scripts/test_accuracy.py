import numpy as np
from PIL import Image
import click
import math

from deblurgan.model import generator_model
from deblurgan.utils import load_images, deprocess_image


def test(batch_size):
    data = load_images('./images/test', batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights('generator.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    acc = 0

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        mse = np.sum((y-img)**(2))/(generated.shape[1]*generated.shape[2]*generated.shape[3])
        psnr = 10*math.log10((255**2)/mse)
        acc = acc + psnr
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))
    
    final_acc = acc/(generated_images.shape[0])
    print ('test accuracy',  final_acc)


@click.command()
@click.option('--batch_size', default=4, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
