import * as tf from '@tensorflow/tfjs-layers';
import * as tfc from '@tensorflow/tfjs-core';
import {MNISTDataset} from '../../src';

function buildDenseModel() {
  const model = tf.sequential();
  model.add(
      tf.layers.dense({units: 42, inputShape: [784], activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function onBodyLoad() {
  document.getElementById('train').addEventListener('click', train);
}

async function train() {
  document.getElementById('train').setAttribute('disabled', 'true');

  const ds = await MNISTDataset.create();
  const model = buildDenseModel();
  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  const batchProgressEl = document.getElementById('batch-progress');
  const epochEndResultEl = document.getElementById('epoch-end-result');

  console.log(await ds.trainDataset.collectAll());

  const fds = ds.trainDataset.map((row: Array<{[key: string]: number}>) => {
    const [rawFeatures, rawLabel] = row;
    const convertedFeatures = Object.values(rawFeatures);
    const convertedLabel = Object.values(rawLabel);
    return [convertedFeatures, convertedLabel];
  }).batch(3);

  console.log(await fds.collectAll());

  await model.fitDataset(fds, {
    epochs: 3,
    callbacks: {
      onBatchEnd: async (batch: number, logs?: tf.Logs) => {
        batchProgressEl.innerText =
            `${batch} - ${logs['loss']} -  ${logs['acc']}`;
      },
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        epochEndResultEl.innerText =
            `${epoch} - ${logs['loss']} -  ${logs['acc']}`;
      }
    }
  });

  document.getElementById('train').setAttribute('disabled', 'false');
}

document.addEventListener('DOMContentLoaded', onBodyLoad, false);
