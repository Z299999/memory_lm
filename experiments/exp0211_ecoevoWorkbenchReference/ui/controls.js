/**
 * Control panel: reads parameters from DOM inputs and exposes them.
 */

export class Controls {
  constructor() {
    // Start-time params (apply on Reset)
    this.elM = document.getElementById('param-m');
    this.elN = document.getElementById('param-n');
    this.elKInternal = document.getElementById('param-k'); // legacy (unused)
    this.elArchitecture = document.getElementById('param-arch');
    this.elInputSource = document.getElementById('param-input-source');
    this.elConstruction = document.getElementById('param-construction');
    this.elActivation = document.getElementById('param-activation');
    this.elWeightControl = document.getElementById('param-weight-control');

    // Run-time params (apply immediately)
    this.elMu = document.getElementById('param-mu');
    this.elMuVal = document.getElementById('param-mu-val');
    this.elPFlip = document.getElementById('param-pflip');
    this.elPFlipVal = document.getElementById('param-pflip-val');
    this.elTBridge = document.getElementById('param-tbridge');
    this.elTBridgeVal = document.getElementById('param-tbridge-val');
    this.elSigma = document.getElementById('param-sigma');
    this.elSigmaVal = document.getElementById('param-sigma-val');
    this.elOmega = document.getElementById('param-omega');
    this.elOmegaVal = document.getElementById('param-omega-val');
    this.elEpsZero = document.getElementById('param-epszero');
    this.elEpsZeroVal = document.getElementById('param-epszero-val');
    this.elPAddEdge = document.getElementById('param-p-add-edge');
    this.elPAddEdgeVal = document.getElementById('param-p-add-edge-val');
    this.elPAddNode = document.getElementById('param-p-add-node');
    this.elPAddNodeVal = document.getElementById('param-p-add-node-val');
    this.elRandAlpha = document.getElementById('param-rand-alpha');
    this.elRandAlphaVal = document.getElementById('param-rand-alpha-val');
    this.elRandDMax = document.getElementById('param-rand-dmax');
    this.elRandDMaxVal = document.getElementById('param-rand-dmax-val');
    this.elTheta = document.getElementById('param-theta');
    this.elThetaVal = document.getElementById('param-theta-val');
    this.elK = document.getElementById('param-K');
    this.elKVal = document.getElementById('param-K-val');
    this.elSpeed = document.getElementById('param-speed');
    this.elSpeedVal = document.getElementById('param-speed-val');
    this.elOuMean = document.getElementById('param-ou-mean');
    this.elOuMeanVal = document.getElementById('param-ou-mean-val');
    this.elEtaHebb = document.getElementById('param-eta-hebb');
    this.elEtaHebbVal = document.getElementById('param-eta-hebb-val');
    this.elHebbThresh = document.getElementById('param-hebb-thresh');
    this.elHebbThreshVal = document.getElementById('param-hebb-thresh-val');

    // Impulse test params
    this.elTestInput = document.getElementById('param-test-input');
    this.elTestAmp = document.getElementById('param-test-amp');
    this.elTestSteps = document.getElementById('param-test-steps');
    this.elTestSignalType = document.getElementById('param-test-signal-type');

    // Show live values next to sliders
    this._bindSliderDisplay(this.elMu, this.elMuVal);
    this._bindSliderDisplay(this.elPFlip, this.elPFlipVal);
    this._bindSliderDisplay(this.elTBridge, this.elTBridgeVal);
    this._bindSliderDisplay(this.elSigma, this.elSigmaVal);
    this._bindSliderDisplay(this.elOmega, this.elOmegaVal);
    this._bindSliderDisplay(this.elEpsZero, this.elEpsZeroVal);
    this._bindSliderDisplay(this.elPAddEdge, this.elPAddEdgeVal);
    this._bindSliderDisplay(this.elPAddNode, this.elPAddNodeVal);
    this._bindSliderDisplay(this.elRandAlpha, this.elRandAlphaVal);
    this._bindSliderDisplay(this.elRandDMax, this.elRandDMaxVal);
    this._bindSliderDisplay(this.elTheta, this.elThetaVal);
    this._bindSliderDisplay(this.elK, this.elKVal);
    this._bindSliderDisplay(this.elSpeed, this.elSpeedVal);
    if (this.elOuMean && this.elOuMeanVal) {
      this._bindSliderDisplay(this.elOuMean, this.elOuMeanVal);
    }
    if (this.elEtaHebb && this.elEtaHebbVal) {
      this._bindSliderDisplay(this.elEtaHebb, this.elEtaHebbVal);
    }
    if (this.elHebbThresh && this.elHebbThreshVal) {
      this._bindSliderDisplay(this.elHebbThresh, this.elHebbThreshVal);
    }
  }

  _bindSliderDisplay(slider, display) {
    const update = () => { display.textContent = slider.value; };
    slider.addEventListener('input', update);
    update();
  }

  /** Get start-time parameters (used on Reset). */
  getStartParams() {
    const mVal = parseInt(this.elM.value, 10);
    const arch = this.elArchitecture?.value || 'core9';
    let kInternal;
    if (arch === 'mlp2') {
      // Two hidden layers, each of size m
      const mSafe = Number.isFinite(mVal) && mVal > 0 ? mVal : 1;
      kInternal = 2 * mSafe;
    } else {
      // 9-node fully connected core (default)
      kInternal = 9;
    }
    return {
      m: mVal,
      n: parseInt(this.elN.value, 10),
      kInternal,
      arch,
      inputSource: this.elInputSource.value,
      activation: this.elActivation?.value || 'tanh',
      weightControl: this.elWeightControl?.value || 'vanilla',
      construction: this.elConstruction?.value || 'bridge'
    };
  }

  /** Get run-time parameters (used every step). */
  getRunParams() {
    return {
      mu: parseFloat(this.elMu.value),
      pFlip: parseFloat(this.elPFlip.value),
      tBridge: parseFloat(this.elTBridge.value),
      sigma: parseFloat(this.elSigma.value),
      omega: parseFloat(this.elOmega.value),
      epsilon: parseFloat(this.elEpsZero.value),
      K: parseInt(this.elK.value, 10),
      inputSource: this.elInputSource.value,
      theta: this.elTheta ? parseFloat(this.elTheta.value) : 0,
      pAddEdge: this.elPAddEdge ? parseFloat(this.elPAddEdge.value) : 0.01,
      pAddNode: this.elPAddNode ? parseFloat(this.elPAddNode.value) : 0.005,
      randAlpha: this.elRandAlpha ? parseFloat(this.elRandAlpha.value) : 1.5,
      randDMax: this.elRandDMax ? parseInt(this.elRandDMax.value, 10) || 4 : 4,
      ouMean: this.elOuMean ? parseFloat(this.elOuMean.value) || 0 : 0,
      etaHebb: this.elEtaHebb ? parseFloat(this.elEtaHebb.value) || 0 : 0,
      hebbThresh: this.elHebbThresh ? parseFloat(this.elHebbThresh.value) || 0 : 0
    };
  }

  /** Get speed in steps per second. */
  getSpeed() {
    return parseInt(this.elSpeed.value, 10);
  }

  /** Get impulse test parameters. */
  getTestParams() {
    return {
      inputIndex: parseInt(this.elTestInput.value, 10) || 0,
      amplitude: parseFloat(this.elTestAmp.value) || 1,
      steps: parseInt(this.elTestSteps.value, 10) || 200,
      signalType: this.elTestSignalType ? this.elTestSignalType.value || 'impulse' : 'impulse'
    };
  }
}
