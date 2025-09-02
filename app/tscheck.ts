// Test strict TypeScript rules
const arr: string[] = [];
const item = arr[0]; // Should trigger noUncheckedIndexedAccess
item.toLowerCase(); // Should error due to potential undefined

// Test useUnknownInCatchVariables
try {
  throw new Error('test');
} catch (e) {
  const message = e.message; // Should require type assertion with useUnknownInCatchVariables
}

// Test exactOptionalPropertyTypes
interface Test {
  name?: string;
}
const test: Test = {};
test.name = undefined; // Should trigger exactOptionalPropertyTypes error
