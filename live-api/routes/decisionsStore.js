// CommonJS version of decisions store for server.js compatibility
const RING_MAX = 5000;
const ring = [];

const decisionsStore = {
  push(evt) {
    ring.push(evt);
    if (ring.length > RING_MAX) ring.shift();
  },

  querySince(since) {
    return ring.filter(event => new Date(event.ts).getTime() >= since);
  }
};

// Export for both CommonJS and ES modules
module.exports = { decisionsStore };
