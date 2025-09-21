module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.spec.ts', '**/tests/**/*.test.ts'],
  collectCoverageFrom: [
    'routes/**/*.ts',
    'api/**/*.ts',
    '!routes/**/*.d.ts',
    '!api/**/*.d.ts'
  ],
  setupFilesAfterEnv: [],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1'
  }
};
