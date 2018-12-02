# Dataset API (tfjs-data) for MNIST

This package provides the Dataset API for MNIST dataset. It is built using [@tensorflow/tfjs-data](https://github.com/tensorflow/tfjs-data) package (which is now
included in [@tensorflow/tfjs](https://github.com/tensorflow/tfjs) union package) that provides a uniform and consistent way to access various datasets.

## Installation

npm install tfjs-data-mnist

## Usage

```ts
// get the dataset
const ds = await MNISTDataset.create();

// there are 2 properties in ds (testDataset and trainDataset)

// get the iterator for testDataset
const it = await ds.testDataset.iterator();

// iterate by invoking next
const dataElement =  await it.next();

// dataElement.done === true => there are no more elements 

// dataElement.value is **TensorContainer** of type [feature, label]
// where feature and label are of type Tensor1D
//
// feature is Tensor1D with shape [784]
// label is Tensor1D with shape [10]
//
//
// label is actually a one-hot encoded vector

// how to get the feature and label
const feature = dataElement.value[0] as tfjs.Tensor;
const label = dataElement.value[1] as tfjs.Tensor;

// The nice thing about dataset API is that you get
// lot of operations such as suffle, repeat, take etc
// for free

// Here is an example to first shuffle the dataset
// and then take only first 5 samples

const shuffled5 = await ds.testDataset.shuffle(10).take(5).iterator();

```

## Example

### Running the sample

```bash

# do npm install at the root of this directory
npm install

# install peer dependnencies
npm install @tensorflow/tfjs-core @tensorflow/tfjs-data --no-save

# change directory into example
cd example

# do npm install in example
npm install

# finally run the example
npm start
```