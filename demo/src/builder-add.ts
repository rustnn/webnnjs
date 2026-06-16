import { MLGraphBuilder, ml } from '@webnnjs/webnn-node';

import { ensureOrtDylibPath } from './ort-env.js';

async function main(): Promise<void> {
  ensureOrtDylibPath();
  if (process.env.ORT_DYLIB_PATH) {
    console.log(`Using ORT_DYLIB_PATH=${process.env.ORT_DYLIB_PATH}`);
  }

  const context = await ml.createContext({ accelerated: true });
  console.log(`Context accelerated: ${context.accelerated}`);

  const builder = new MLGraphBuilder(context);
  const a = builder.input('a', { dataType: 'float32', shape: [2, 2] });
  const b = builder.input('b', { dataType: 'float32', shape: [2, 2] });
  const out = builder.add(a, b);
  const graph = await builder.build({ result: out });

  const tensorDesc = {
    dataType: 'float32' as const,
    shape: [2, 2],
    readable: true,
    writable: true,
  };

  const tensorA = await context.createTensor(tensorDesc);
  const tensorB = await context.createTensor(tensorDesc);
  const tensorOut = await context.createTensor(tensorDesc);

  const valuesA = new Float32Array([1, 2, 3, 4]);
  const valuesB = new Float32Array([5, 6, 7, 8]);

  context.writeTensor(tensorA, valuesA);
  context.writeTensor(tensorB, valuesB);

  context.dispatch(graph, { a: tensorA, b: tensorB }, { result: tensorOut });

  const outputBytes = await context.readTensor(tensorOut);
  const result = new Float32Array(outputBytes);

  console.log('Input A:', [...valuesA]);
  console.log('Input B:', [...valuesB]);
  console.log('A + B =  ', [...result]);

  tensorA.destroy();
  tensorB.destroy();
  tensorOut.destroy();
  graph.destroy();
  context.destroy();
}

main().catch((error) => {
  console.error('Builder demo failed:', error);
  process.exitCode = 1;
});
