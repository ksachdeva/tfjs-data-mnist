import {MNISTDataset} from '../../src';

async function onBodyLoad() {
  const ds = await MNISTDataset.create();
  const tensors = await ds.testDataset.iterator();

  const val = await tensors.next();
  console.log(val);
  console.log(await tensors.next());

  const ds2 = ds.testDataset.batch(2);

  const tensors2 = await ds2.iterator();

  const val2 = await tensors2.next();
  console.log(val2);
  console.log(await tensors2.next());
}

document.addEventListener('DOMContentLoaded', onBodyLoad, false);