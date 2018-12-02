import * as tfjs from '@tensorflow/tfjs-core';
import * as tfd from '@tensorflow/tfjs-data';

import {MNISTDataset} from '../../src';

let ds: MNISTDataset = null;

async function toPixels(sample: tfd.DataElement, element: HTMLCanvasElement) {
  const reshapedTensor = sample[0].reshape([28, 28, 1]);
  await tfjs.toPixels(reshapedTensor, element);
}

async function onBodyLoad() {
  document.getElementById('shuffledButton')
      .addEventListener('click', onShuffleImages);

  ds = await MNISTDataset.create();

  // we take the first 5 elements
  const only5 = await ds.testDataset.take(5).iterator();

  let n = await only5.next();
  let i = 1;

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
  const shuffled5 = await ds.testDataset.shuffle(10).take(5).iterator();

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