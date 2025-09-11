/**
 * Add missing proof-related columns to fills_snapshot table
 * Required for observability endpoints to work properly
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

async function addProofColumns() {
  const dbPath = path.join(__dirname, '../data/evotester.db');
  const db = new sqlite3.Database(dbPath);

  console.log('Adding proof columns to fills_snapshot table...');

  try {
    // Add overall_passed column to track proof validation results
    await run(db, `
      ALTER TABLE fills_snapshot
      ADD COLUMN overall_passed INTEGER DEFAULT 0
    `);

    // Add proof_details column to store proof validation details
    await run(db, `
      ALTER TABLE fills_snapshot
      ADD COLUMN proof_details TEXT
    `);

    // Add greeks_reserved column to store pre-trade greeks
    await run(db, `
      ALTER TABLE fills_snapshot
      ADD COLUMN greeks_reserved TEXT
    `);

    // Add greeks_actual column to store post-trade greeks
    await run(db, `
      ALTER TABLE fills_snapshot
      ADD COLUMN greeks_actual TEXT
    `);

    console.log('✅ Successfully added proof columns to fills_snapshot table');

    // Create indexes for the new columns
    await run(db, 'CREATE INDEX IF NOT EXISTS idx_fills_overall_passed ON fills_snapshot(overall_passed)');
    await run(db, 'CREATE INDEX IF NOT EXISTS idx_fills_proof_details ON fills_snapshot(proof_details)');

    console.log('✅ Successfully created indexes for proof columns');

  } catch (error) {
    if (error.message.includes('duplicate column name')) {
      console.log('ℹ️  Proof columns already exist, skipping...');
    } else {
      console.error('❌ Error adding proof columns:', error);
      throw error;
    }
  } finally {
    db.close();
  }
}

function run(db, sql) {
  return new Promise((resolve, reject) => {
    db.run(sql, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

// Run if called directly
if (require.main === module) {
  addProofColumns()
    .then(() => {
      console.log('🎉 Proof columns migration completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('💥 Proof columns migration failed:', error);
      process.exit(1);
    });
}

module.exports = { addProofColumns };
