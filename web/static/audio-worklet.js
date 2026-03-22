class ChunkCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.bufferSize = 256;
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
