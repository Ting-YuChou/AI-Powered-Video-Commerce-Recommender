module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  ignorePatterns: [
    'dist/',
    'node_modules/',
    'frontend_*.js',
    'frontend_*.tsx',
    '*_config.js',
  ],
  rules: {
    'no-unused-vars': ['error', { argsIgnorePattern: '^_', ignoreRestSiblings: true }],
    'react/prop-types': 'off',
    'react/react-in-jsx-scope': 'off',
  },
};
