// Test ESLint rule enforcement
const s = 'test';
s.charAt(0); // Should trigger error for charAt

const n = 1.234;
n.toFixed(2); // Should trigger error for toFixed
