/**
 * Input signal generators for the simulation.
 */

/** Uniform(-1, 1) noise. */
function noiseGenerator(_i, _t) {
  return Math.random() * 2 - 1;
}

/** Constant input: always 1. */
function constantGenerator(_i, _t) {
  return 1;
}

/** Sine wave with varied frequency per input index. */
function sineGenerator(i, t) {
  const period = 100;
  const freq = 1 + i * 0.3;  // varied frequency per input
  const phase = i * 0.7;     // varied phase per input
  return Math.sin(2 * Math.PI * freq * t / period + phase);
}

const generators = {
  noise: noiseGenerator,
  constant: constantGenerator,
  sine: sineGenerator
};

/** Get the input value for input node index i at step t. */
export function getInputValue(source, i, t) {
  const gen = generators[source];
  if (!gen) throw new Error(`Unknown input source: ${source}`);
  return gen(i, t);
}
