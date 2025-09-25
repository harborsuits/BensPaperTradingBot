#!/usr/bin/env node

/**
 * Manual control script for testing trading components
 * Use this to start/stop AutoLoop and BrainIntegrator manually
 */

const fetch = require('node-fetch');
const readline = require('readline');

const API_BASE = 'http://localhost:4000';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function startAutoLoop() {
  try {
    const response = await fetch(`${API_BASE}/api/manual/start-autoloop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();
    console.log('AutoLoop start:', result);
  } catch (error) {
    console.error('Failed to start AutoLoop:', error.message);
  }
}

async function stopAutoLoop() {
  try {
    const response = await fetch(`${API_BASE}/api/manual/stop-autoloop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();
    console.log('AutoLoop stop:', result);
  } catch (error) {
    console.error('Failed to stop AutoLoop:', error.message);
  }
}

async function getStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/manual/status`);
    const status = await response.json();
    console.log('\n=== System Status ===');
    console.log('AutoLoop:', status.autoLoop);
    console.log('BrainIntegrator:', status.brainIntegrator);
    console.log('AI Orchestrator:', status.aiOrchestrator);
    console.log('Market Open:', status.marketOpen);
    console.log('Positions:', status.positions);
    console.log('==================\n');
  } catch (error) {
    console.error('Failed to get status:', error.message);
  }
}

async function testBrainScore(symbol) {
  try {
    const response = await fetch(`${API_BASE}/api/brain/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    });
    const score = await response.json();
    console.log(`\nBrain score for ${symbol}:`, score.final_score || score.score);
    console.log('Confidence:', score.conf || 'N/A');
  } catch (error) {
    console.error('Failed to get brain score:', error.message);
  }
}

function showMenu() {
  console.log('\n=== Trading Control Menu ===');
  console.log('1. Start AutoLoop (override market hours)');
  console.log('2. Stop AutoLoop');
  console.log('3. Get System Status');
  console.log('4. Test Brain Score');
  console.log('5. Exit');
  console.log('========================\n');
}

async function handleChoice(choice) {
  switch (choice.trim()) {
    case '1':
      await startAutoLoop();
      break;
    case '2':
      await stopAutoLoop();
      break;
    case '3':
      await getStatus();
      break;
    case '4':
      rl.question('Enter symbol to test: ', async (symbol) => {
        await testBrainScore(symbol.toUpperCase());
        promptUser();
      });
      return; // Don't prompt again here
    case '5':
      console.log('Exiting...');
      process.exit(0);
    default:
      console.log('Invalid choice. Please try again.');
  }
  promptUser();
}

function promptUser() {
  showMenu();
  rl.question('Enter your choice: ', handleChoice);
}

// Start the interactive menu
console.log('Trading System Manual Control');
promptUser();

