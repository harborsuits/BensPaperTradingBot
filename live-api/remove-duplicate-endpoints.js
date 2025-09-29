#!/usr/bin/env node

// This script removes duplicate endpoints that were accidentally added

const fs = require('fs');
const path = require('path');

const serverFile = path.join(__dirname, 'minimal_server.js');
const content = fs.readFileSync(serverFile, 'utf8');

// Find the section we added
const startMarker = '// ========== EVOLUTION & DISCOVERY ENDPOINTS ==========';
const endMarker = '// Removed duplicate poolStatus endpoint';

const startIndex = content.indexOf(startMarker);
const endIndex = content.indexOf(endMarker);

if (startIndex === -1 || endIndex === -1) {
  console.error('Could not find the duplicate section');
  process.exit(1);
}

// Remove the duplicate section
const newContent = content.slice(0, startIndex) + content.slice(endIndex);

// Write back
fs.writeFileSync(serverFile, newContent);

console.log('✅ Removed duplicate endpoints from minimal_server.js');
