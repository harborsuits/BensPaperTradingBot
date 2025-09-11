/**
 * Production Safety Checker
 * Build-time validation to ensure no mocks/fakes in production code
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class ProductionSafetyChecker {
  constructor() {
    this.issues = [];
    this.forbiddenPatterns = [
      { pattern: /Math\.random\(/g, severity: 'error', description: 'Random data generation in production' },
      { pattern: /FAKE/g, severity: 'error', description: 'Fake data placeholder' },
      { pattern: /mock/g, severity: 'warning', description: 'Mock data usage' },
      { pattern: /synthetic/g, severity: 'warning', description: 'Synthetic data generation' },
      { pattern: /fake/g, severity: 'warning', description: 'Fake data indicator' },
      { pattern: /stub/g, severity: 'warning', description: 'Stub implementation' },
      { pattern: /dummy/g, severity: 'warning', description: 'Dummy data' },
      { pattern: /console\.log\(/g, severity: 'warning', description: 'Debug logging in production' },
      { pattern: /debugger/g, severity: 'error', description: 'Debugger statement' }
    ];

    this.allowedFiles = [
      '**/test/**',
      '**/tests/**',
      '**/spec/**',
      '**/mocks/**',
      '**/fixtures/**',
      '**/node_modules/**',
      'synthetic.ts',
      'synthetic.js',
      'mockAPI.js',
      'api_test.html',
      'check-production-safety.js',
      'real-data-status.js',
      'setup-real-data-mode.sh'
    ];

    this.allowedDirectories = [
      'scripts/',
      'docs/',
      'examples/'
    ];
  }

  /**
   * Check if file should be excluded from production safety checks
   */
  isAllowedFile(filePath) {
    const relativePath = path.relative(process.cwd(), filePath);

    // Check exact file matches
    if (this.allowedFiles.some(pattern => relativePath.includes(pattern))) {
      return true;
    }

    // Check directory matches
    if (this.allowedDirectories.some(dir => relativePath.startsWith(dir))) {
      return true;
    }

    // Allow files in test/spec directories
    if (relativePath.includes('/test/') || relativePath.includes('/tests/') ||
        relativePath.includes('/spec/') || relativePath.includes('/mocks/')) {
      return true;
    }

    return false;
  }

  /**
   * Scan file for forbidden patterns
   */
  scanFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');

      lines.forEach((line, index) => {
        this.forbiddenPatterns.forEach(({ pattern, severity, description }) => {
          const matches = line.match(pattern);
          if (matches) {
            this.issues.push({
              file: filePath,
              line: index + 1,
              pattern: pattern.source,
              description: description,
              content: line.trim(),
              severity: severity,
              match: matches[0]
            });
          }
        });
      });
    } catch (error) {
      this.issues.push({
        file: filePath,
        line: 0,
        pattern: 'FILE_READ_ERROR',
        description: 'File read error',
        content: error.message,
        severity: 'warning'
      });
    }
  }

  /**
   * Scan directory recursively
   */
  scanDirectory(dirPath) {
    const items = fs.readdirSync(dirPath);

    items.forEach(item => {
      const fullPath = path.join(dirPath, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        // Skip node_modules and other build artifacts
        if (!['node_modules', '.git', 'dist', 'build'].includes(item)) {
          this.scanDirectory(fullPath);
        }
      } else if (stat.isFile()) {
        // Only check JavaScript/TypeScript files
        if (['.js', '.ts', '.jsx', '.tsx'].includes(path.extname(item))) {
          if (!this.isAllowedFile(fullPath)) {
            this.scanFile(fullPath);
          }
        }
      }
    });
  }

  /**
   * Check runtime configuration for production mode
   */
  checkRuntimeConfig() {
    try {
      const configPath = path.join(__dirname, '../src/services/runtimeConfig.js');
      const content = fs.readFileSync(configPath, 'utf8');

      if (!content.includes('DATA_MODE: \'real_only\'')) {
        this.issues.push({
          file: configPath,
          line: 0,
          pattern: 'INCORRECT_DATA_MODE',
          content: 'Runtime config must have DATA_MODE: \'real_only\'',
          severity: 'error'
        });
      }
    } catch (error) {
      this.issues.push({
        file: 'runtimeConfig.js',
        line: 0,
        pattern: 'CONFIG_MISSING',
        content: 'Cannot find runtime configuration file',
        severity: 'error'
      });
    }
  }

  /**
   * Run all production safety checks
   */
  async runChecks() {
    console.log('🔍 Running production safety checks...\n');

    // Check runtime configuration
    this.checkRuntimeConfig();

    // Scan source code
    const srcDir = path.join(__dirname, '../src');
    if (fs.existsSync(srcDir)) {
      this.scanDirectory(srcDir);
    }

    // Scan root level JS files
    const rootFiles = fs.readdirSync(path.join(__dirname, '..')).filter(f =>
      ['.js', '.ts'].includes(path.extname(f)) && f !== 'package-lock.json'
    );

    rootFiles.forEach(file => {
      const filePath = path.join(__dirname, '..', file);
      if (!this.isAllowedFile(filePath)) {
        this.scanFile(filePath);
      }
    });

    return this.issues;
  }

  /**
   * Report issues found
   */
  reportIssues() {
    if (this.issues.length === 0) {
      console.log('✅ No production safety issues found!');
      return true;
    }

    const errors = this.issues.filter(i => i.severity === 'error');
    const warnings = this.issues.filter(i => i.severity === 'warning');

    console.log(`❌ Found ${this.issues.length} issues:`);
    console.log(`   Errors: ${errors.length}`);
    console.log(`   Warnings: ${warnings.length}\n`);

    // Group by file for cleaner output
    const byFile = {};
    this.issues.forEach(issue => {
      if (!byFile[issue.file]) byFile[issue.file] = [];
      byFile[issue.file].push(issue);
    });

    Object.entries(byFile).forEach(([file, fileIssues]) => {
      console.log(`📁 ${file}:`);
      fileIssues.forEach(issue => {
        const icon = issue.severity === 'error' ? '❌' : '⚠️';
        console.log(`  ${icon} Line ${issue.line}: ${issue.description}`);
        console.log(`     Pattern: ${issue.pattern}`);
        if (issue.match) console.log(`     Match: ${issue.match}`);
        console.log(`     Content: ${issue.content}\n`);
      });
    });

    return errors.length === 0;
  }

  /**
   * Generate CI-compatible output
   */
  generateCIOutput() {
    const errors = this.issues.filter(i => i.severity === 'error');

    if (errors.length === 0) {
      console.log('✅ CI: No blocking issues found');
      return;
    }

    console.log(`❌ CI: Found ${errors.length} blocking issues`);

    // GitHub Actions format
    errors.forEach(issue => {
      console.log(`::error file=${issue.file},line=${issue.line}::${issue.description} - ${issue.pattern}`);
    });

    // Exit with error code for CI
    process.exit(1);
  }

  /**
   * Exit with appropriate code based on issues
   */
  exitWithCode() {
    const hasErrors = this.issues.some(i => i.severity === 'error');
    process.exit(hasErrors ? 1 : 0);
  }
}

// Run checks if called directly
if (require.main === module) {
  const checker = new ProductionSafetyChecker();
  const isCI = process.env.CI === 'true' || process.argv.includes('--ci');

  checker.runChecks()
    .then(() => {
      if (isCI) {
        checker.generateCIOutput();
      } else {
        const passed = checker.reportIssues();
        if (!passed) {
          console.log('🚫 Production safety checks FAILED - fix issues before deploying');
          process.exit(1);
        } else {
          console.log('🎉 Production safety checks PASSED');
          process.exit(0);
        }
      }
    })
    .catch(error => {
      console.error('Error running production safety checks:', error);
      process.exit(1);
    });
}

module.exports = { ProductionSafetyChecker };
