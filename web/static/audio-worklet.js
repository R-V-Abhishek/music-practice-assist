class ChunkCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = [];
    const requested = options?.processorOptions?.bufferSize;
    // Low-latency default: 512 samples (~10.7 ms @ 48 kHz)
    this.bufferSize = Number.isFinite(requested) && requested >= 128 ? Math.floor(requested) : 512;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }

    const channelData = input[0];
    for (let i = 0; i < channelData.length; i++) {
      this.buffer.push(channelData[i]);
      if (this.buffer.length >= this.bufferSize) {
        const chunk = this.buffer.slice(0, this.bufferSize);
        this.buffer = this.buffer.slice(this.bufferSize);
        this.port.postMessage(new Float32Array(chunk));
      }
    }

    return true;
  }
}

registerProcessor("chunk-capture-processor", ChunkCaptureProcessor);
