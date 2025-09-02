import { test, expect } from '@playwright/test';

test('quick list renders and navigates', async ({ page }) => {
  await page.route('/api/scanner/candidates', route =>
    route.fulfill({ 
      json: { 
        candidates: [
          { symbol: 'PLTR', score: 0.9 }, 
          { symbol: 'AAPL', score: 0.7 }
        ] 
      } 
    })
  );
  
  await page.goto('/');
  
  // Check that the quick list is visible
  await expect(page.getByRole('list', { name: /candidates quick list/i })).toBeVisible();
  
  // Click on a candidate
  await page.getByRole('button', { name: /open decisions for PLTR/i }).click();
  
  // Check that we navigated to the decisions page with the correct symbol
  await expect(page).toHaveURL(/decisions\?symbol=PLTR/);
});

test('candidate card shows data from API', async ({ page }) => {
  // Mock the API response with a timestamp
  await page.route('/api/scanner/candidates', route =>
    route.fulfill({ 
      json: { 
        candidates: [
          { 
            symbol: 'TSLA', 
            score: 0.85,
            confidence: 0.75,
            side: 'buy',
            last: 242.5,
            asOf: new Date().toISOString()
          }
        ],
        asOf: new Date().toISOString()
      } 
    })
  );
  
  await page.goto('/');
  
  // Check that the candidate card shows the correct symbol
  await expect(page.getByRole('heading', { name: 'TSLA' })).toBeVisible();
  
  // Check that the timestamp is shown
  await expect(page.getByText(/as of \d{2}:\d{2}:\d{2}/)).toBeVisible();
  
  // Check that the "Trace & Decisions" button is present
  await expect(page.getByRole('button', { name: /view decisions for TSLA/i })).toBeVisible();
});

test('empty state is shown when no candidates', async ({ page }) => {
  // Mock empty candidates response
  await page.route('/api/scanner/candidates', route =>
    route.fulfill({ json: { candidates: [] } })
  );
  
  await page.goto('/');
  
  // Check that the empty state is shown
  await expect(page.getByText('No candidates')).toBeVisible();
});
