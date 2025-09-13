/**
 * Seed Generator Service
 *
 * Generates strategy phenotypes for EvoTester based on family configurations
 * and market regime-specific gene bounds.
 */

class SeedGenerator {
    constructor(policy) {
        this.policy = policy;
        this.families = policy.families || {};
    }

    /**
     * Generate phenotypes for EvoTester seeding
     */
    generatePhenotypes(request) {
        const { families, count, bounds, objective } = request;
        const phenotypes = [];

        console.log(`[SEED_GENERATOR] Generating ${count} phenotypes for families:`, families.map(f => f.name));

        for (let i = 0; i < count; i++) {
            // Select family based on weights
            const selectedFamily = this.selectWeightedFamily(families);

            // Generate parameters for this family
            const params = this.generateFamilyParameters(selectedFamily, bounds);

            // Create phenotype object
            const phenotype = {
                id: `evo_${Date.now()}_${i}_${Math.random().toString(36).substr(2, 9)}`,
                label: `${selectedFamily.name}_phenotype_${i}`,
                family: selectedFamily.name,
                params: params,
                constraints: this.getFamilyConstraints(selectedFamily.name),
                generation: 1,
                fitness: null,
                metadata: {
                    objective: objective || 'after_cost_sharpe_pf_dd',
                    created_at: new Date().toISOString(),
                    family_weight: selectedFamily.weight
                }
            };

            phenotypes.push(phenotype);
        }

        console.log(`[SEED_GENERATOR] Generated ${phenotypes.length} phenotypes`);
        return phenotypes;
    }

    /**
     * Select family based on weights
     */
    selectWeightedFamily(families) {
        const totalWeight = families.reduce((sum, f) => sum + (f.weight || 1), 0);
        let random = Math.random() * totalWeight;

        for (const family of families) {
            random -= family.weight || 1;
            if (random <= 0) {
                return family;
            }
        }

        return families[0]; // fallback
    }

    /**
     * Generate parameters for a specific family
     */
    generateFamilyParameters(family, customBounds = {}) {
        const familyName = family.name;
        const familyConfig = this.families[familyName];

        if (!familyConfig) {
            console.warn(`[SEED_GENERATOR] Unknown family: ${familyName}`);
            return {};
        }

        const genes = familyConfig.genes || {};
        const params = {};

        // Use custom bounds if provided, otherwise use family defaults
        const bounds = customBounds[familyName] || genes;

        Object.entries(bounds).forEach(([paramName, range]) => {
            if (Array.isArray(range) && range.length === 2) {
                const [min, max] = range;

                if (typeof min === 'number' && typeof max === 'number') {
                    // Numeric range - generate random value
                    params[paramName] = this.generateNumericParameter(min, max, paramName);
                } else if (typeof min === 'string' && typeof max === 'string') {
                    // Time range - generate random time
                    params[paramName] = this.generateTimeParameter(min, max);
                } else {
                    // Discrete options - pick random
                    params[paramName] = this.generateDiscreteParameter(range);
                }
            } else if (Array.isArray(range)) {
                // Discrete options
                params[paramName] = this.generateDiscreteParameter(range);
            } else {
                // Single value
                params[paramName] = range;
            }
        });

        return params;
    }

    /**
     * Generate numeric parameter within range
     */
    generateNumericParameter(min, max, paramName = '') {
        // Add some intelligence based on parameter type
        let value;

        if (paramName.includes('pct') || paramName.includes('percent')) {
            // Percentage parameters - use more conservative distribution
            value = min + Math.random() * (max - min) * 0.8; // Favor lower end
        } else if (paramName.includes('atr') || paramName.includes('stop')) {
            // Risk parameters - use normal distribution
            const mean = (min + max) / 2;
            const std = (max - min) / 6; // 99.7% within range
            value = this.normalRandom(mean, std);
            value = Math.max(min, Math.min(max, value));
        } else {
            // General parameters - uniform distribution
            value = min + Math.random() * (max - min);
        }

        // Round appropriately
        if (max - min < 1) {
            return Math.round(value * 100) / 100; // 2 decimal places
        } else if (max - min < 10) {
            return Math.round(value * 10) / 10; // 1 decimal place
        } else {
            return Math.round(value); // Integer
        }
    }

    /**
     * Generate time parameter
     */
    generateTimeParameter(minTime, maxTime) {
        // Parse time strings like "09:35" and "11:00"
        const parseTime = (timeStr) => {
            const [hours, minutes] = timeStr.split(':').map(Number);
            return hours * 60 + minutes; // Convert to minutes since midnight
        };

        const minMinutes = parseTime(minTime);
        const maxMinutes = parseTime(maxTime);

        const randomMinutes = minMinutes + Math.random() * (maxMinutes - minMinutes);
        const hours = Math.floor(randomMinutes / 60);
        const minutes = Math.floor(randomMinutes % 60);

        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    }

    /**
     * Generate discrete parameter from options
     */
    generateDiscreteParameter(options) {
        if (!Array.isArray(options) || options.length === 0) {
            return null;
        }

        // Equal probability for discrete options
        const randomIndex = Math.floor(Math.random() * options.length);
        return options[randomIndex];
    }

    /**
     * Get constraints for a family
     */
    getFamilyConstraints(familyName) {
        const familyConfig = this.families[familyName];
        if (!familyConfig) return {};

        // Return default constraints - can be extended based on family
        return {
            max_position_size: 0.1, // 10% of capital
            max_daily_trades: 10,
            min_holding_period: 1, // minutes
            max_holding_period: 1440, // minutes (1 day)
            risk_per_trade: 0.02 // 2% risk per trade
        };
    }

    /**
     * Generate normally distributed random number
     */
    normalRandom(mean, std) {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();

        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z0 * std + mean;
    }

    /**
     * Validate phenotype parameters
     */
    validatePhenotype(phenotype) {
        const errors = [];

        // Basic validation
        if (!phenotype.id) errors.push('Missing phenotype ID');
        if (!phenotype.family) errors.push('Missing family');
        if (!phenotype.params) errors.push('Missing parameters');

        // Family-specific validation
        if (phenotype.family === 'trend') {
            this.validateTrendParameters(phenotype.params, errors);
        } else if (phenotype.family === 'meanrev') {
            this.validateMeanRevParameters(phenotype.params, errors);
        } else if (phenotype.family === 'breakout') {
            this.validateBreakoutParameters(phenotype.params, errors);
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    /**
     * Validate trend family parameters
     */
    validateTrendParameters(params, errors) {
        if (params.ma_fast >= params.ma_slow) {
            errors.push('Fast MA must be less than slow MA');
        }

        if (params.adx_min < 10 || params.adx_min > 40) {
            errors.push('ADX minimum should be between 10-40');
        }

        if (params.sl_atr < 0.5 || params.sl_atr > 3.0) {
            errors.push('Stop loss ATR should be between 0.5-3.0');
        }
    }

    /**
     * Validate mean reversion family parameters
     */
    validateMeanRevParameters(params, errors) {
        if (params.rsi_buy >= params.rsi_sell) {
            errors.push('RSI buy must be less than RSI sell');
        }

        if (params.rsi_buy < 10 || params.rsi_sell > 90) {
            errors.push('RSI levels should be between 10-90');
        }

        if (params.bb_dev < 1.0 || params.bb_dev > 3.0) {
            errors.push('Bollinger deviation should be between 1.0-3.0');
        }
    }

    /**
     * Validate breakout family parameters
     */
    validateBreakoutParameters(params, errors) {
        if (params.range_mins < 1 || params.range_mins > 60) {
            errors.push('Range minutes should be between 1-60');
        }

        if (params.min_vol_x < 1.0 || params.min_vol_x > 5.0) {
            errors.push('Min volume multiplier should be between 1.0-5.0');
        }
    }

    /**
     * Get family statistics for monitoring
     */
    getFamilyStats() {
        const stats = {};

        Object.keys(this.families).forEach(familyName => {
            const family = this.families[familyName];
            stats[familyName] = {
                gene_count: Object.keys(family.genes || {}).length,
                weight: family.weight || 0,
                parameters: Object.keys(family.genes || {})
            };
        });

        return stats;
    }
}

module.exports = SeedGenerator;
