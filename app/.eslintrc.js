module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  plugins: ['react', '@typescript-eslint'],
  settings: {
    react: {
      version: 'detect',
    },
  },
  rules: {
    // Ban dangerous string/number methods
    'no-restricted-syntax': [
      'error',
      { 
        'selector': 'MemberExpression[property.name=\'charAt\']', 
        'message': 'Use utils/string.initial() or capitalize().' 
      },
      { 
        'selector': 'MemberExpression[property.name=\'toFixed\']', 
        'message': 'Use utils/number.fmtNum()/fmtPercent().' 
      },
      {
        'selector': 'CallExpression[callee.property.name=\'toFixed\']',
        'message': 'Use utils/number.fmtNum()/fmtPercent() helpers.'
      }
    ],
    
    // Additional protection for prototype methods
    'no-restricted-properties': [
      'error',
      { 
        'object': 'String', 
        'property': 'charAt', 
        'message': 'Use utils/string.initial() or capitalize().'
      },
      { 
        'property': 'toFixed', 
        'message': 'Use fmtNum/fmtPercent helpers.'
      }
    ],
    'react/react-in-jsx-scope': 'off', // Not needed with newer React versions
    'react-hooks/exhaustive-deps': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': ['warn', { 'argsIgnorePattern': '^_' }],
  },
};
