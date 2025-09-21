'use strict';

module.exports = function provenance(tag) {
  return function provenanceMiddleware(_req, res, next) {
    try {
      if (tag) res.setHeader('x-provenance', tag);
    } catch {}
    next();
  };
};


