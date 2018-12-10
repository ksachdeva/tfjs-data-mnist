import * as tf from '@tensorflow/tfjs';

import {MNISTDataset} from '../../src';

let ds: MNISTDataset = null;

async function toPixels(sample: number[][], element: HTMLCanvasElement) {
  const reshapedTensor = tf.tensor(sample[0]).reshape([28, 28, 1]);
  await tf.toPixels(reshapedTensor as tf.Tensor2D, element);
}

async function onBodyLoad() {
  document.getElementById('shuffledButton')
      .addEventListener('click', onShuffleImages);

  ds = await MNISTDataset.create();

  // we take the first 5 elements
  const only5 = await ds.testDataset
                    .map((row: Array<{[key: string]: number}>) => {
                      const [rawFeatures, rawLabel] = row;
                      const convertedFeatures = Object.values(rawFeatures);
                      const convertedLabel = Object.values(rawLabel);
                      return [convertedFeatures, convertedLabel];
                    })
                    .take(5)
                    .iterator();

  let n = await only5.next();
  let i = 1;

  console.log(n);

  const feature = tf.tensor(n.value[0]);
  const label = tf.tensor(n.value[1]);

  document.getElementById('feature-shape').innerText = feature.shape.toString();
  document.getElementById('label-shape').innerText = label.shape.toString();

  do {
    await toPixels(
        n.value, document.getElementById('canvas' + i) as HTMLCanvasElement);
    n = await only5.next();
    i = i + 1;
  } while (!n.done);

  onShuffleImages();
}

async function onShuffleImages() {
  // we take shuffle the dataset and then take 5 elements
  const shuffled5 = await ds.testDataset
                        .map((row: Array<{[key: string]: number}>) => {
                          const [rawFeatures, rawLabel] = row;
                          const convertedFeatures = Object.values(rawFeatures);
                          const convertedLabel = Object.values(rawLabel);
                          return [convertedFeatures, convertedLabel];
                        })
                        .shuffle(10)
                        .take(5)
                        .iterator();

  let n = await shuffled5.next();
  let i = 6;

  do {
    await toPixels(
        n.value, document.getElementById('canvas' + i) as HTMLCanvasElement);
    n = await shuffled5.next();
    i = i + 1;
  } while (!n.done);
}

document.addEventListener('DOMContentLoaded', onBodyLoad, false);
